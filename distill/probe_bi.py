"""Block Influence: which blocks barely change what passes through them.

ShortGPT's metric, verbatim:

    BI_i = 1 - E_{x,t} [ cos( x_in[t], x_out[t] ) ]

averaged over non-padding tokens of a calibration set. A block whose output is
nearly its input is a pass-through and can be deleted; a high-BI block is doing
work. One forward pass scores every block at once.

This complements probe_depth.py rather than replacing it. Truncation can only
remove a suffix of the stack; BI finds dead blocks in the MIDDLE. Run the depth
probe first -- it bounds how much depth is removable at all -- then use this to
choose WHICH blocks go.

CALIBRATION DATA MATTERS. ShortGPT scores on generic pretraining text. Here the
default is the downstream benchmark molecules, because that is the distribution
the pruned model has to survive on, and the teacher's behaviour already varies
several-fold across these benchmarks (KL 0.117 on THPep vs 0.419 on CellPPD).

A caveat worth keeping in view: BI is a LOCAL criterion. It measures how much a
block changes the residual stream, not how much the final pooled vector -- the
only thing used downstream -- depends on it. Blocks can be individually
low-influence and jointly load-bearing, so the ranking is a proposal to be
checked by actually exporting and evaluating, not a verdict.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from student import patch_sdpa_dtype

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(R, "their_repo", "data")

SOURCES = {
    "AmpHGT": (D + "/amp_train.csv", "smiles"),
    "CellPPD": (D + "/CellPPD_train.csv", "smiles"),
    "THPep": (D + "/THPep_main90_smiles_classes.csv", "smiles"),
    "PAMPA": (D + "/PAMPA_clusters.csv", "SMILES"),
    "PepMSND": (D + "/PepMSND_clustered_data.csv", "SMILES"),
}


def load_calibration(names, n_per, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for name in names:
        path, col = SOURCES[name]
        s = pd.read_csv(path)[col].dropna().astype(str).unique()
        take = s if len(s) <= n_per else rng.choice(s, n_per, replace=False)
        out += list(take)
        print("   %-8s %d molecules" % (name, len(take)))
    return out


@torch.no_grad()
def block_influence(model, tok, smiles, device, batch=8, max_length=512):
    blocks = model.model.transformer.blocks
    nb = len(blocks)
    num = torch.zeros(nb, dtype=torch.float64)
    den = torch.zeros(nb, dtype=torch.float64)
    state = {"mask": None}

    def mk(i):
        def hook(mod, inp, out):
            # inp[0] is the block's input, out is its output: exactly the pair
            # BI is defined over.
            c = F.cosine_similarity(inp[0].float(), out.float(), dim=-1)
            m = state["mask"]
            num[i] += (c * m).sum().double().cpu()
            den[i] += m.sum().double().cpu()
        return hook

    hooks = [b.register_forward_hook(mk(i)) for i, b in enumerate(blocks)]
    for i in range(0, len(smiles), batch):
        enc = tok(smiles[i:i + batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length)
        ids = enc["input_ids"].to(device)
        att = enc["attention_mask"].to(device)
        state["mask"] = (ids != 0).float()
        with torch.autocast(device.split(":")[0], dtype=torch.float16,
                            enabled=(device != "cpu")):
            model(input_ids=ids, attention_mask=att)
        if i and (i // batch) % 100 == 0:
            print("   %d/%d" % (i, len(smiles)), flush=True)
    for h in hooks:
        h.remove()
    return (1.0 - (num / den)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--calib", default="AmpHGT,PAMPA,THPep",
                    help="benchmarks to draw calibration molecules from")
    ap.add_argument("--n-per", type=int, default=400)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    src = a.model or glob.glob(
        R + "/models/models--aaronfeller--peptideclm-2-mlm-large/snapshots/*")[0]
    patch_sdpa_dtype()
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    model = AutoModel.from_pretrained(src, trust_remote_code=True,
                                      use_safetensors=True).to(a.device).eval()
    nb = len(model.model.transformer.blocks)

    print("calibration:")
    smis = load_calibration(a.calib.split(","), a.n_per)
    print("   total %d molecules" % len(smis))

    bi = block_influence(model, tok, smis, a.device, a.batch, a.max_length)
    order = np.argsort(bi)

    print("\n%-7s %10s   %s" % ("block", "BI", "rank (1 = most removable)"))
    rank = {int(b): r + 1 for r, b in enumerate(order)}
    for i in range(nb):
        bar = "#" * int(round(60 * bi[i] / max(bi.max(), 1e-9)))
        print("%-7d %10.5f   %-4d %s" % (i, bi[i], rank[i], bar))

    print("\nlowest-BI blocks, in removal order:")
    for k in (2, 4, 6, 8, 12, 16):
        if k <= nb:
            drop = sorted(int(x) for x in order[:k])
            print("   drop %2d (%.0f%% of blocks): %s" % (k, 100 * k / nb, drop))
            print("      export_truncated.py --out <dir> --drop %s"
                  % ",".join(map(str, drop)))

    out = a.out or os.path.join(R, "results", "pruning", "block_influence.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"model": os.path.basename(src), "calib": a.calib,
               "n_molecules": len(smis), "bi": [float(x) for x in bi],
               "removal_order": [int(x) for x in order]}, open(out, "w"), indent=1)
    print("\nwrote " + out)


if __name__ == "__main__":
    main()
