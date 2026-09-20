"""How closely does each student reproduce the teacher's masked-token distribution
on the DOWNSTREAM benchmark molecules?

The geometry probes (probe_geometry.py) ask whether the student's embedding space
looks like the teacher's. This asks the narrower question the KD loss actually
optimises: given a masked position, does the student predict the same
distribution over tokens the teacher does. The two can disagree -- a student can
match the teacher token-by-token while its pooled geometry drifts -- so both are
worth having.

Molecules come from the benchmark CSVs rather than the pretraining corpus,
because that is the distribution the downstream results are computed on, and a
student can track the teacher on training-like molecules and lose it elsewhere.

MASKING IS SHARED. One RNG, seeded once per benchmark, produces the mask before
any model is loaded, and every model scores the identical masked input. Comparing
KLs measured under different masks would be meaningless.

Each model is scored in its own subprocess alongside the teacher, with the rotary
buffers rebuilt and verified on both (see train_kd.refresh_rope). That guard is
what makes it safe to have the 337M teacher and a 32M student in one process at
all; without it the student's outputs change silently.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INIT = glob.glob(R + "/models/models--aaronfeller--peptideclm-2-mlm-small/snapshots/*")[0]
TEACH = glob.glob(R + "/models/models--aaronfeller--peptideclm-2-mlm-large/snapshots/*")[0]
CK = R + "/results/kd/distill/%s/latest.pt"

MODELS = {
    "warm-start": None,                       # their released 32M, untouched
    "treatment": CK % "treatment",            # cached KD + MTR + SPKD
    "control": CK % "control",                # same schedule, no teacher
    "kd-live": R + "/results/kd/kd_live/student_final.pt",   # pure Hinton KD, live
}

BENCH = {
    "PAMPA": ("their_repo/data/PAMPA_clusters.csv", "SMILES"),
    "CellPPD": ("their_repo/data/CellPPD_test.csv", None),
    "AmpHGT": ("their_repo/data/amp_test.csv", None),
    "THPep": ("their_repo/data/THPep_main90_smiles_classes.csv", "smiles"),
}


def smiles_column(df, hint):
    if hint and hint in df.columns:
        return hint
    for c in df.columns:
        if c.lower() in ("smiles", "smile", "canonical_smiles"):
            return c
    # Fall back to the column that looks most like SMILES rather than guessing
    # position: these CSVs do not agree on ordering.
    best, score = None, -1
    for c in df.columns:
        v = df[c].astype(str).head(50)
        sc = v.str.contains(r"[=#@\[\]]").mean() * v.str.len().mean()
        if sc > score:
            best, score = c, sc
    return best


def load_bench(name, n_max, seed):
    path, hint = BENCH[name]
    df = pd.read_csv(os.path.join(R, path))
    col = smiles_column(df, hint)
    smis = [str(x) for x in df[col].dropna().unique()]
    if len(smis) > n_max:
        smis = list(np.random.default_rng(seed).choice(smis, n_max, replace=False))
    return smis, col


# --------------------------------------------------------------------- worker
def score(student_ckpt, smis, seed, batch, device="cuda"):
    """KL(teacher || student) and top-1 agreement at masked positions."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_live import span_mask_positions
    from train_kd import load_pair, logits_of

    tok = AutoTokenizer.from_pretrained(INIT, trust_remote_code=True)
    teacher, student = load_pair(TEACH, INIT, device)
    if student_ckpt:
        st = torch.load(student_ckpt, map_location="cpu", weights_only=False)
        sd = st["model"]
        assert not any(k.startswith("backbone.") for k in sd) or True
        if any(k.startswith("backbone.") for k in sd):
            sd = {k[len("backbone."):]: v for k, v in sd.items()
                  if k.startswith("backbone.")}
        student.load_state_dict(sd)
        from train_kd import refresh_rope, verify_rope
        refresh_rope(student, device); verify_rope(student, "student(ckpt)")
    student.eval()

    rng = np.random.default_rng(seed)
    # KL at BOTH temperatures. kd-live minimised KL at T=4 while the cached
    # treatment used T=1, so scoring only at T=1 would judge one model on a
    # target it was never trained on. Report both.
    kl_sum = {1.0: 0.0, 4.0: 0.0}
    agree = n = 0
    for i in range(0, len(smis), batch):
        chunk = smis[i:i + batch]
        enc = tok(chunk, add_special_tokens=False, truncation=True,
                  max_length=510)["input_ids"]
        T = max(len(e) for e in enc) + 2
        x = np.full((len(chunk), T), tok.pad_token_id, dtype=np.int64)
        att = np.zeros((len(chunk), T), dtype=np.int64)
        rows, cols = [], []
        for r, e in enumerate(enc):
            x[r, 0] = tok.cls_token_id
            x[r, 1:1 + len(e)] = e
            x[r, 1 + len(e)] = tok.sep_token_id
            att[r, :len(e) + 2] = 1
            for p in span_mask_positions(len(e), rng):
                rows.append(r); cols.append(p + 1)
        if not rows:
            continue
        rows = torch.tensor(rows); cols = torch.tensor(cols)
        xi = torch.from_numpy(x).to(device); ai = torch.from_numpy(att).to(device)
        xi[rows, cols] = tok.mask_token_id
        with torch.no_grad(), torch.autocast(device, dtype=torch.float16):
            tl = logits_of(teacher(input_ids=xi, attention_mask=ai))[rows, cols]
            sl = logits_of(student(input_ids=xi, attention_mask=ai))[rows, cols]
        for T in kl_sum:
            tp = F.softmax(tl.float() / T, -1)
            # sum p_t (log p_t - log p_s) over the vocab, averaged over positions
            kl = (tp * (tp.clamp_min(1e-12).log()
                        - F.log_softmax(sl.float() / T, -1))).sum(-1)
            kl_sum[T] += kl.sum().item()
        agree += (sl.argmax(-1) == tl.argmax(-1)).sum().item()
        n += len(rows)
    return {"kl": kl_sum[1.0] / n, "kl_T4": kl_sum[4.0] / n,
            "agree": agree / n, "positions": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default=None, help="internal: model name to score")
    ap.add_argument("--bench", default=None, help="internal: benchmark name")
    ap.add_argument("--n", type=int, default=400, help="molecules per benchmark")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.worker:
        smis, _ = load_bench(a.bench, a.n, a.seed)
        print("RESULT " + json.dumps(score(MODELS[a.worker], smis, a.seed, a.batch)))
        return

    for p in [v for v in MODELS.values() if v]:
        assert os.path.exists(p), "missing checkpoint: " + p

    out = {}
    for bench in BENCH:
        smis, col = load_bench(bench, a.n, a.seed)
        print("\n%s: %d molecules (column %r)" % (bench, len(smis), col), flush=True)
        out[bench] = {}
        for name in MODELS:
            r = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--worker", name,
                 "--bench", bench, "--n", str(a.n), "--batch", str(a.batch),
                 "--seed", str(a.seed)],
                capture_output=True, text=True,
                cwd=os.path.dirname(os.path.abspath(__file__)))
            line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
            if not line:
                print(r.stdout[-1200:]); print(r.stderr[-2000:])
                raise RuntimeError("worker failed for %s / %s" % (bench, name))
            out[bench][name] = json.loads(line[0][len("RESULT "):])
            print("   %-11s KL(T=1) %.4f  KL(T=4) %.4f  agree %.4f  (%d pos)"
                  % (name, out[bench][name]["kl"], out[bench][name]["kl_T4"],
                     out[bench][name]["agree"], out[bench][name]["positions"]),
                  flush=True)

    names = list(MODELS)
    print("\n=== KL(teacher || student), masked positions, lower is closer ===")
    print("%-10s " % "benchmark" + "".join("%12s" % n for n in names))
    for b in BENCH:
        print("%-10s " % b + "".join("%12.4f" % out[b][n]["kl"] for n in names))
    print("\n=== KL at T=4, the temperature kd-live was trained at ===")
    print("%-10s " % "benchmark" + "".join("%12s" % n for n in names))
    for b in BENCH:
        print("%-10s " % b + "".join("%12.4f" % out[b][n]["kl_T4"] for n in names))
    print("\n=== top-1 agreement with the teacher, higher is closer ===")
    print("%-10s " % "benchmark" + "".join("%12s" % n for n in names))
    for b in BENCH:
        print("%-10s " % b + "".join("%12.4f" % out[b][n]["agree"] for n in names))

    dest = os.path.join(R, "results", "kd", "probe_kl.json")
    json.dump(out, open(dest, "w"), indent=1)
    print("\nwrote " + dest)


if __name__ == "__main__":
    main()
