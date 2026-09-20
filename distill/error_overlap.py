"""Do the 337M teacher and the 84.8M pruned student fail on the SAME molecules?

Run on every benchmark where both models scored the IDENTICAL molecules:

    AmpHGT    5118 unique molecules, single split
    CellPPD    291 unique molecules, 5-fold ensembled
    PepMSND    640 molecules, 10 folds pooled

THPep is excluded and cannot be added: their shipped predictions come from their
own split, ours from a split we generated, so the two contain different molecules
and there is nothing to pair.

A molecule counts as an error for a model if it is misclassified in a MAJORITY of
that model's three runs. Per-seed errors would conflate model behaviour with seed
noise; majority voting keeps only molecules that are reliably wrong.

Writes results/analysis/error_overlap/error_overlap_<bench>.csv, one row per molecule with both models'
scores and error flags, for the chemistry pass in error_chemistry.py.

TWO DATA TRAPS, both of which silently produced wrong answers before being found:

  amp_test.csv has 5148 rows but 5118 unique SMILES -- 30 molecules appear twice.
      Labels agree and there is no train/test leakage, so duplicates are collapsed
      by averaging rather than dropped.

  their PepMSND rerun files PREFIX the molecule key with the run name
      ("kan_rerun2_fold10_idx0" vs "fold10_idx0"). Joined naively, the three runs
      share no keys at all and the majority vote reports zero teacher errors.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PUB = os.path.join(R, "their_repo", "figure_generation", "results")
DATA = os.path.join(R, "their_repo", "data")


def from_results_csv(path, key="smiles"):
    """Their classification output. 5-fold files repeat the test set once per
    fold; pool by mean logit, the aggregation reverse-engineered from their
    shipped CellPPD predictions."""
    d = pd.read_csv(path)
    if "fold" in d.columns and d.fold.nunique() > 1:
        d["i"] = d.groupby("fold").cumcount()
        g = d.groupby("i")
        d = pd.DataFrame({key: g[key].first().values,
                          "true": g.true_label.first().values,
                          "score": g.predicted_label.mean().values})
    else:
        d = d[[key, "true_label", "predicted_label"]].rename(
            columns={"true_label": "true", "predicted_label": "score"})
    return d.groupby(key).agg(true=("true", "first"), score=("score", "mean"))


def majority(frames, threshold):
    """Per-molecule error flag by majority vote across runs, plus mean score."""
    errs, scores = [], []
    for i, f in enumerate(frames):
        errs.append((((f.score > threshold).astype(int) != f.true.astype(int))
                     .astype(int)).rename(i))
        scores.append(f.score.rename(i))
    W = pd.concat(errs, axis=1)
    S = pd.concat(scores, axis=1)
    return pd.DataFrame({"true": frames[0].true.astype(int),
                         "err": (W.sum(axis=1) >= (len(frames) + 1) // 2).astype(int),
                         "score": S.mean(axis=1)})


def report(name, T, S, threshold):
    idx = T.index.intersection(S.index)
    T, S = T.loc[idx], S.loc[idx]
    assert (T.true.values == S.true.values).all(), "labels disagree after join"
    y = T.true.values.astype(int)
    te, se = T.err.values, S.err.values
    both = int(((te == 1) & (se == 1)).sum())
    exp = te.mean() * se.mean() * len(y)

    print("=" * 78)
    print("%s -- %d molecules, %d positive (%.0f%%)" % (name, len(y), y.sum(),
                                                        100 * y.mean()))
    print("=" * 78)
    print("   teacher wrong  %4d (%.1f%%) | student wrong %4d (%.1f%%)"
          % (te.sum(), 100 * te.mean(), se.sum(), 100 * se.mean()))
    print("   BOTH wrong     %4d   vs %.1f expected if independent  -> %.1fx enrichment"
          % (both, exp, both / max(exp, 1e-9)))
    print("   teacher only   %4d | student only %4d | neither %4d"
          % (((te == 1) & (se == 0)).sum(), ((te == 0) & (se == 1)).sum(),
             ((te == 0) & (se == 0)).sum()))
    print("   Cohen kappa %.3f | %.0f%% of teacher errors shared | %.0f%% of student errors shared"
          % (cohen_kappa_score(te, se), 100 * both / max(te.sum(), 1),
             100 * both / max(se.sum(), 1)))
    print("   %-6s teacher %4d  student %4d  both %4d  (%.0f%% of teacher's)"
          % ("FP", te[y == 0].sum(), se[y == 0].sum(),
             int(((te == 1) & (se == 1) & (y == 0)).sum()),
             100 * ((te == 1) & (se == 1) & (y == 0)).sum() / max(te[y == 0].sum(), 1)))
    print("   %-6s teacher %4d  student %4d  both %4d  (%.0f%% of teacher's)"
          % ("FN", te[y == 1].sum(), se[y == 1].sum(),
             int(((te == 1) & (se == 1) & (y == 1)).sum()),
             100 * ((te == 1) & (se == 1) & (y == 1)).sum() / max(te[y == 1].sum(), 1)))

    out = pd.DataFrame({"smiles": idx, "true": y,
                        "teacher_score": T.score.values, "student_score": S.score.values,
                        "teacher_err": te, "student_err": se})
    out["err_type"] = np.where(out.true == 1, "FN", "FP")
    out.loc[(out.teacher_err == 0) & (out.student_err == 0), "err_type"] = "-"
    return out


# ---------------------------------------------------------------- AmpHGT, CellPPD
for bench, tdir, sglob, thr in (
        ("amphgt", "amp_hgt", ["results/pruning/bi_confirm/bisel8/seed_101/*_results.csv",
                               "results/pruning/bisel8_sweep/amphgt/seed_*/*_results.csv"], 0.0),
        ("cellppd", "cellppd", ["results/pruning/bisel8_cls/CellPPD/seed_*/*_results.csv"], 0.0)):
    tp = sorted(glob.glob(os.path.join(
        PUB, "runs_LoRA_highrank/%s/peptideclm-2-mlm-large/seed_*/*_results.csv" % tdir)))
    sp = []
    for g in sglob:
        sp += sorted(glob.glob(os.path.join(R, g)))
    if len(tp) != 3 or len(sp) != 3:
        print("SKIP %s: teacher %d, student %d files" % (bench, len(tp), len(sp)))
        continue
    T = majority([from_results_csv(p) for p in tp], thr)
    S = majority([from_results_csv(p) for p in sp], thr)
    out = report(bench.upper(), T, S, thr)
    out.to_csv(os.path.join(R, "results", "analysis", "error_overlap", "error_overlap_%s.csv" % bench), index=False)
    print()

# ---------------------------------------------------------------- PepMSND
smi = {}
for k in range(1, 11):
    x = pd.read_csv(os.path.join(DATA, "PepMSND_data", "X_test%d.csv" % k))
    for i_, sm in enumerate(x.SMILES.values):
        smi["fold%d_idx%d" % (k, i_)] = sm

tf = []
for p in sorted(glob.glob(os.path.join(PUB, "PepMSND_results_final/kan_*_all_predictions.csv"))):
    d = pd.read_csv(p)
    d = d[d.model.str.contains("PeptideMLM_lg", na=False)]
    d = d.assign(molecule=d.molecule.str.replace(r"^kan_rerun\d+_", "", regex=True))
    g = d.set_index("molecule")
    tf.append(pd.DataFrame({"true": g.true.astype(int), "score": g.predicted}))

sf = []
for seed in (101, 202, 303):
    parts = []
    for k in range(1, 11):
        f = R + "/results/pruning/bisel8_sweep/pepmsnd/seed_%d/fold_%d/preds_fold%d.csv" % (seed, k, k)
        if not os.path.exists(f):
            parts = []
            break
        p_ = pd.read_csv(f)
        p_["molecule"] = ["fold%d_idx%d" % (k, i_) for i_ in range(len(p_))]
        parts.append(p_)
    if parts:
        d = pd.concat(parts, ignore_index=True).set_index("molecule")
        sf.append(pd.DataFrame({"true": d.true_label.astype(int), "score": d.predicted_prob}))

if len(tf) == 3 and len(sf) == 3:
    out = report("PEPMSND", majority(tf, 0.5), majority(sf, 0.5), 0.5)
    out["smiles"] = [smi[m] for m in out.smiles]      # key -> SMILES for chemistry
    out.to_csv(os.path.join(R, "results", "analysis", "error_overlap", "error_overlap_pepmsnd.csv"), index=False)
else:
    print("PepMSND skipped: teacher %d runs, student %d seeds" % (len(tf), len(sf)))

print()
for b in ("amphgt", "cellppd", "pepmsnd"):
    p = os.path.join(R, "results", "analysis", "error_overlap", "error_overlap_%s.csv" % b)
    if os.path.exists(p):
        print("wrote %s" % p)
