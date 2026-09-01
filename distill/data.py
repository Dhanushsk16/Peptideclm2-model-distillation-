"""Streaming loader over the teacher cache.

One training example is one molecule:
    input   the SMILES, tokenized and masked at exactly the positions the teacher
            saw -- the cache is only valid for that exact input
    targets teacher top-16 log-probs at those positions   -> KD
            the true token ids at those positions          -> MLM (control arm)
            99 pre-normalized RDKit descriptors            -> MTR
            teacher mean_pool from the CLEAN forward       -> SPKD

Design note: this streams SHARD BY SHARD rather than randomly indexing molecules.
np.load returns a lazy NpzFile whose __getitem__ re-reads the WHOLE array from
disk on every access, so per-molecule random access would reload ~64 MB per
lookup. Materialising one shard at a time costs ~230 MB resident and turns that
into a single sequential read. Randomness is not lost: the corpus was shuffled
before sharding, so each shard is already an unbiased sample, and we shuffle
shard order and batch order on top.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import torch


class SubsetTable:
    """SMILES + descriptors for the whole subset, held once."""

    def __init__(self, parquet_path):
        df = pd.read_parquet(parquet_path)
        drop = {"source", "split", "smiles", "n_tokens", "__index_level_0__"}
        self.desc_cols = [c for c in df.columns if c not in drop]
        assert len(self.desc_cols) == 99, \
            "expected 99 descriptors, got %d" % len(self.desc_cols)
        self.smiles = df["smiles"].to_numpy()
        # fp16 halves the resident cost (800 MB -> 400 MB) and the targets are
        # z-scored with |x| < 600, far inside fp16 range.
        self.desc = df[self.desc_cols].to_numpy(dtype=np.float16)


def build_batches(lengths, max_tokens, rng=None):
    """Length-bucketed batches under a token budget.

    The corpus is bimodal -- PubChem ~23 tokens, peptides ~340 -- so fixed-size
    batches would pad short molecules ~20x. Sorting by length and filling to a
    token budget keeps batches dense and memory predictable, since cost scales
    with B*T rather than B.
    """
    order = np.argsort(lengths, kind="stable")
    batches, cur, curmax = [], [], 0
    for i in order:
        L = int(lengths[i])
        if cur and max(curmax, L) * (len(cur) + 1) > max_tokens:
            batches.append(cur)
            cur, curmax = [int(i)], L
        else:
            cur.append(int(i))
            curmax = max(curmax, L)
    if cur:
        batches.append(cur)
    if rng is not None:
        rng.shuffle(batches)
    return batches


class ShardStream:
    """Iterates batches over all shards for one epoch."""

    def __init__(self, cache_dir, table, tokenizer, mask="a", max_tokens=16384,
                 seed=0, shard_limit=None):
        self.paths = sorted(glob.glob(os.path.join(cache_dir, "shard_*.npz")))
        if not self.paths:
            raise FileNotFoundError("no shard_*.npz under " + cache_dir)
        if shard_limit:
            self.paths = self.paths[:shard_limit]
        self.table = table
        self.tok = tokenizer
        self.mask = mask
        self.max_tokens = max_tokens
        self.seed = seed
        self.CLS = tokenizer.cls_token_id
        self.SEP = tokenizer.sep_token_id
        self.PAD = tokenizer.pad_token_id
        self.MASK = tokenizer.mask_token_id

    def _load_shard(self, path):
        """Pull the arrays we need into RAM once (~230 MB for a 50k shard)."""
        with np.load(path) as z:
            m = self.mask
            return {
                "mol_idx": z["mol_idx"],
                "mean_pool": z["mean_pool"],
                "ptr": z["ptr_" + m],
                "pos": z["pos_" + m],
                "topi": z["topi_" + m],
                "topv": z["topv_" + m],
            }

    def epoch_batches(self, epoch):
        """(path, [molecule rows]) for every batch, so a run can be resumed at a
        step boundary without re-deriving the whole schedule."""
        rng = np.random.default_rng(self.seed + epoch)
        paths = list(self.paths)
        rng.shuffle(paths)
        return paths, rng

    def __iter__(self):
        yield from self.iter_epoch(0)

    def iter_epoch(self, epoch, skip_batches=0):
        paths, rng = self.epoch_batches(epoch)
        seen = 0
        for path in paths:
            sh = self._load_shard(path)
            smiles = self.table.smiles[sh["mol_idx"]]
            desc = self.table.desc[sh["mol_idx"]]
            ids = self.tok(list(smiles), add_special_tokens=False)["input_ids"]
            lengths = np.array([len(x) + 2 for x in ids])

            batches = build_batches(lengths, self.max_tokens, rng)
            for rows in batches:
                if seen < skip_batches:            # fast-forward on resume
                    seen += 1
                    continue
                seen += 1
                yield self._collate(rows, ids, lengths, desc, sh)
            del sh, ids

    def _collate(self, rows, ids, lengths, desc, sh):
        T = int(lengths[rows].max())
        B = len(rows)
        x = np.full((B, T), self.PAD, dtype=np.int64)
        att = np.zeros((B, T), dtype=np.int64)
        for r, i in enumerate(rows):
            s = ids[i]
            x[r, 0] = self.CLS
            x[r, 1:1 + len(s)] = s
            x[r, 1 + len(s)] = self.SEP
            att[r, :len(s) + 2] = 1

        mrow, mcol, topi, topv = [], [], [], []
        for r, i in enumerate(rows):
            lo, hi = sh["ptr"][i], sh["ptr"][i + 1]
            p = sh["pos"][lo:hi].astype(np.int64)
            mrow.append(np.full(len(p), r, dtype=np.int64))
            mcol.append(p)
            topi.append(sh["topi"][lo:hi])
            topv.append(sh["topv"][lo:hi])
        mrow = np.concatenate(mrow) if mrow else np.zeros(0, np.int64)
        mcol = np.concatenate(mcol) if mcol else np.zeros(0, np.int64)

        # True token ids BEFORE masking -- the hard labels the control arm needs.
        labels = x[mrow, mcol].copy()
        x[mrow, mcol] = self.MASK

        return {
            "input_ids": torch.from_numpy(x),
            "attention_mask": torch.from_numpy(att),
            "rows": torch.from_numpy(mrow),
            "cols": torch.from_numpy(mcol),
            "labels": torch.from_numpy(labels),
            "topi": torch.from_numpy(np.concatenate(topi).astype(np.int64)),
            "topv": torch.from_numpy(np.concatenate(topv).astype(np.float32)),
            "teacher_pool": torch.from_numpy(sh["mean_pool"][rows].astype(np.float32)),
            "desc": torch.from_numpy(desc[rows].astype(np.float32)),
        }
