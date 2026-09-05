"""Data for live-teacher distillation: SMILES in, masked batches out.

No teacher cache is involved, so this is much simpler than data.py -- there are no
shards to align against and no precomputed targets to keep in step. The only
outputs are the masked input, the positions that were masked, and the true tokens
that were there.

Because the teacher runs live, the mask is re-sampled EVERY EPOCH. The cached
pipeline could not do that: cached logits are only valid for the mask they were
computed under, so it reused one fixed mask per molecule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

MASK_PCT, SPAN_MU, SPAN_SD = 0.25, 3.5, 1.0


def span_mask_positions(seq_len, rng, guard_mult=50):
    """Verbatim port of their pretraining.py:202-233.

    Per-sequence 25% budget, span lengths ~ N(3.5, 1.0) floored at 1, spans
    truncated so the budget lands exactly, and an overlap check that rejects a
    span by collapsing it to zero length.

    ONE DELIBERATE DEVIATION: their while-loop has no escape. On a short, crowded
    sequence no non-overlapping span can be placed and it spins forever. The guard
    stops early instead.
    """
    n_target = int(seq_len * MASK_PCT)
    masked, count, guard = set(), 0, 0
    while count < n_target and guard < guard_mult * max(1, n_target):
        guard += 1
        span = max(1, int(rng.normal(SPAN_MU, SPAN_SD)))
        if count + span > n_target:
            span = n_target - count
        start = int(rng.integers(0, seq_len))
        end = min(seq_len, start + span)
        for p in range(start, end):
            if p in masked:
                end = start
                break
        masked.update(range(start, end))
        count += (end - start)
    return sorted(masked)


def build_batches(lengths, max_tokens, rng=None):
    """Length-bucketed batches under a token budget.

    The corpus is bimodal -- PubChem ~23 tokens, ESMAtlas peptides ~340 -- so
    fixed-size batches pad short molecules ~20x. Sorting by length and filling to
    a token budget keeps batches dense and memory predictable, since cost scales
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


class LiveStream:
    """Streams masked batches over the 2M-molecule subset.

    Works in chunks rather than loading every tokenized sequence at once: 320M
    tokens as Python int lists would be several GB, while one 50k chunk is a few
    hundred MB and is discarded after use.
    """

    def __init__(self, parquet_path, tokenizer, max_tokens=8192, chunk=50_000,
                 seed=0, val_molecules=20_000, split="train", limit=None):
        df = pd.read_parquet(parquet_path, columns=["smiles"])
        smiles = df["smiles"].to_numpy()
        if limit:
            smiles = smiles[:limit]

        # Held-out tail, never trained on. Used to watch whether the student is
        # tracking the teacher on molecules it has not seen this epoch.
        if split == "train":
            self.smiles = smiles[:-val_molecules] if val_molecules else smiles
        elif split == "val":
            self.smiles = smiles[-val_molecules:]
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.tok = tokenizer
        self.max_tokens = max_tokens
        self.chunk = chunk
        self.seed = seed
        self.CLS = tokenizer.cls_token_id
        self.SEP = tokenizer.sep_token_id
        self.PAD = tokenizer.pad_token_id
        self.MASK = tokenizer.mask_token_id

    def __len__(self):
        return len(self.smiles)

    def iter_epoch(self, epoch, skip_batches=0):
        """Yield batches for one epoch. The mask RNG is keyed on (seed, epoch),
        so every epoch sees different masked positions but a resumed run
        reproduces the same ones."""
        rng = np.random.default_rng(self.seed * 1000 + epoch)
        starts = list(range(0, len(self.smiles), self.chunk))
        rng.shuffle(starts)

        seen = 0
        for s0 in starts:
            block = list(self.smiles[s0:s0 + self.chunk])
            enc = self.tok(block, add_special_tokens=False)["input_ids"]
            lengths = np.array([len(x) + 2 for x in enc])
            for rows in build_batches(lengths, self.max_tokens, rng):
                if seen < skip_batches:
                    seen += 1
                    continue
                seen += 1
                out = self._collate(rows, enc, lengths, rng)
                if out is not None:
                    yield out
            del enc, block

    def _collate(self, rows, enc, lengths, rng):
        T = int(lengths[rows].max())
        B = len(rows)
        x = np.full((B, T), self.PAD, dtype=np.int64)
        att = np.zeros((B, T), dtype=np.int64)
        for r, i in enumerate(rows):
            s = enc[i]
            x[r, 0] = self.CLS
            x[r, 1:1 + len(s)] = s
            x[r, 1 + len(s)] = self.SEP
            att[r, :len(s) + 2] = 1

        mrow, mcol = [], []
        for r, i in enumerate(rows):
            # Mask in pre-special coordinates, then shift by +1 for the [CLS] we
            # prepended -- so [CLS] and [SEP] are never masked, matching their
            # pretraining, which masked before adding special tokens.
            for p in span_mask_positions(len(enc[i]), rng):
                mrow.append(r)
                mcol.append(p + 1)
        if not mrow:
            return None

        mrow = np.asarray(mrow, dtype=np.int64)
        mcol = np.asarray(mcol, dtype=np.int64)
        labels = x[mrow, mcol].copy()      # true tokens, captured BEFORE masking
        x[mrow, mcol] = self.MASK

        return {
            "input_ids": torch.from_numpy(x),
            "attention_mask": torch.from_numpy(att),
            "rows": torch.from_numpy(mrow),
            "cols": torch.from_numpy(mcol),
            "labels": torch.from_numpy(labels),
        }
