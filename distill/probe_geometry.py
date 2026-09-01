"""Geometry probes, each model embedded in its OWN process.

Driver: builds the molecule sets, spawns one probe_embed.py per model, then scores
the saved vectors. Nothing here loads a model, so no cross-model contamination is
possible.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy.stats as st
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TMP = os.path.join(R, "tmp_probe")
os.makedirs(TMP, exist_ok=True)
INIT = glob.glob(R + "/models/models--aaronfeller--peptideclm-2-mlm-small/snapshots/*")[0]
TEACH = glob.glob(R + "/models/models--aaronfeller--peptideclm-2-mlm-large/snapshots/*")[0]
CK = R + "/results/results/distill/%s/latest.pt"

# (extra CLI args for probe_embed.py) -- teacher is a plain HF dir, students need
# --init and optionally --ckpt.
MODELS = {
    "teacher":    ["--model", TEACH],
    "warm-start": ["--init", INIT],
    "treatment":  ["--init", INIT, "--ckpt", CK % "treatment"],
    "control":    ["--init", INIT, "--ckpt", CK % "control"],
}


def embed(tag, model_args, smis, batch=16):
    sp = os.path.join(TMP, tag + "_smiles.npy")
    op = os.path.join(TMP, tag + ".npy")
    np.save(sp, np.array(smis, dtype=object))
    if not os.path.exists(op):
        r = subprocess.run([sys.executable, "probe_embed.py"] + list(model_args) +
                           ["--smiles-npy", sp, "--out", op, "--tokenizer", INIT,
                            "--batch", str(batch)],
                           cwd=os.path.dirname(os.path.abspath(__file__)),
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-1500:]); print(r.stderr[-2500:])
            raise RuntimeError("embed failed for " + tag)
    return torch.tensor(np.load(op))


def cos(x):
    n = F.normalize(x, dim=1); return (n @ n.t()).numpy()


def gram(x):
    return F.normalize(x @ x.t(), p=2, dim=1)


# ---------------------------------------------------------------- molecule sets
d = pd.read_parquet(R + "/data-20260829T131646Z-1-001/data/pretrain_subset_2M/"
                        "pretrain_subset_2M.parquet", columns=["source", "smiles", "n_tokens"])
z = np.load(R + "/cache/shard_00040.npz"); mi = z["mol_idx"]

N = 1024
ok = np.where(d.n_tokens.values[mi] < 400)[0]
pick = np.random.default_rng(0).choice(ok, N, replace=False)
SMIS = list(d.smiles.values[mi[pick]])
T_cached = torch.tensor(z["mean_pool"][pick].astype(np.float32))
iu = np.triu_indices(N, 1)

Z = {k: embed("geo_" + k, v, SMIS) for k, v in MODELS.items()}

# The cached teacher vectors came from the caching notebook; the live ones from a
# clean process. They must agree, or the cache itself is suspect.
ct_cache, ct_live = cos(T_cached), cos(Z["teacher"])
print("teacher cached vs live: max |diff| %.2e | cross-cos %.4f vs %.4f"
      % (np.abs(ct_cache - ct_live).max(), ct_cache[iu].mean(), ct_live[iu].mean()))
ct = ct_live
t = ct[iu]

print("\n=== GEOMETRY vs teacher (%d molecules, %d pairs) ===" % (N, len(iu[0])))
print("%-12s %10s %11s %11s %10s %11s"
      % ("model", "SPKD", "RAW|dcos|", "CENTERED", "Spearman", "cross-cos"))
print("%-12s %10s %11s %11s %10s %11.4f" % ("teacher", "-", "-", "-", "-", t.mean()))
for k in ("warm-start", "treatment", "control"):
    C = cos(Z[k]); s = C[iu]
    print("%-12s %10.3e %11.4f %11.4f %10.4f %11.4f"
          % (k, (gram(Z[k]) - gram(Z["teacher"])).pow(2).mean().item(),
             np.abs(s - t).mean(),
             np.abs((s - s.mean()) - (t - t.mean())).mean(),
             st.spearmanr(s, t).statistic, s.mean()))

# ---------------------------------------------------------------- respelling
def respell(s, n=8):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    canon = Chem.MolToSmiles(m); out = [canon]; tries = 0
    while len(out) < n and tries < 300:
        tries += 1
        v = Chem.MolToSmiles(m, doRandom=True, canonical=False)
        if v not in out: out.append(v)
    if len(out) < n: return None
    return out if all(Chem.MolToSmiles(Chem.MolFromSmiles(v)) == canon for v in out) else None

mols = []
for s in d[d.source.str.startswith("ESM") & (d.n_tokens < 200)].smiles.iloc[:120]:
    v = respell(s)
    if v: mols.append(v)
    if len(mols) == 40: break
flat = [s for v in mols for s in v]
lab = np.repeat(np.arange(len(mols)), 8)
ZR = {k: embed("res_" + k, v, flat) for k, v in MODELS.items()}

print("\n=== RESPELLING INVARIANCE (%d peptides x 8 spellings) ===" % len(mols))
print("%-12s %9s %9s %9s" % ("model", "self", "cross", "MARGIN"))
per = {}
for k in ("teacher", "warm-start", "treatment", "control"):
    C = cos(ZR[k])
    same = lab[:, None] == lab[None, :]; off = ~np.eye(len(flat), dtype=bool)
    sc, cc = C[same & off].mean(), C[~same].mean()
    per[k] = np.array([C[np.ix_(np.where(lab == i)[0], np.where(lab == i)[0])]
                       [np.triu_indices(8, 1)].mean() for i in range(len(mols))])
    print("%-12s %9.4f %9.4f %9.4f" % (k, sc, cc, sc - cc))

print("\ntreatment - control, per-molecule self-cos: %+.4f (paired t p=%.2g)"
      % ((per["treatment"] - per["control"]).mean(),
         st.ttest_rel(per["treatment"], per["control"]).pvalue))
print("treatment - warm-start:                    %+.4f (p=%.2g)"
      % ((per["treatment"] - per["warm-start"]).mean(),
         st.ttest_rel(per["treatment"], per["warm-start"]).pvalue))
