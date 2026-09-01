"""PAMPA evaluation driver.

Runs the AUTHORS' finetune_ensemble.py unmodified, one (test_fold, val_fold) job
at a time through its internal worker entry point, then ensembles and scores.

Why the worker entry point: their top-level loop does
    fold_ids = sorted(data_frame["fold"].unique())
and runs EVERY fold. There is no flag to restrict it. But the script already
exposes --worker_test_fold / --worker_val_fold / --worker_output_path (marked
SUPPRESS) which trains exactly one model and writes its test predictions. That
gives per-fold control without touching their code.

Cluster -> fold mapping is theirs: sorted(unique clusters) enumerated, so with
clusters 1..6 present, cluster 1 -> fold 0 and cluster 6 -> fold 5.

PAMPA is FULL finetuning -- the whole backbone trains, unlike the LoRA used for
the classification benchmarks. Their resolve_* helpers pick batch size and LR by
substring on the model name, which is why the exported folders must say "-small".
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def ensure_shim(repo_root):
    """finetune_ensemble.py imports training.adapters.common and
    training.experiment.manifest, but in the released repo those directories sit
    under training/02_classification_benchmarks_training_code/. Same stale-layout
    problem as run_experiment.py. Copy them up one level and add __init__.py.

    The location also matters for correctness, not just importability: manifest.py
    computes REPO_ROOT as parents[2] of its own path, which only resolves to the
    repo root from training/experiment/. From its shipped location it would point
    at training/ instead.
    """
    import pathlib, shutil
    src = os.path.join(repo_root, "training", "02_classification_benchmarks_training_code")
    for sub in ("adapters", "experiment"):
        dst = os.path.join(repo_root, "training", sub)
        if not os.path.isdir(dst):
            shutil.copytree(os.path.join(src, sub), dst)
    for p in ("training", "training/adapters", "training/experiment"):
        pathlib.Path(os.path.join(repo_root, p, "__init__.py")).touch()


def load_frame(repo_root, script_dir, data_csv):
    """Use THEIR normalizer so row order matches what the worker writes."""
    ensure_shim(repo_root)
    sys.path.insert(0, repo_root)
    sys.path.insert(0, script_dir)
    import finetune_ensemble as fe
    return fe.normalize_perm_frame(pd.read_csv(data_csv)), fe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True, help="their_repo")
    ap.add_argument("--script", required=True, help="path to finetune_ensemble.py")
    ap.add_argument("--data-csv", required=True, help="PAMPA_clusters.csv")
    ap.add_argument("--models", nargs="+", required=True, help="exported model dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--clusters", nargs="+", type=int, default=[1, 6])
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--split-by", choices=("model", "cluster"), default="model",
                    help="what to pin to a GPU. 'model' runs each variant on its "
                         "own card (both clusters), which keeps the arms on "
                         "identical hardware and finishes them together.")
    ap.add_argument("--max-epochs", type=int, default=250)
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--patience", type=int, default=20)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    script_dir = os.path.dirname(os.path.abspath(a.script))
    repo_root = os.path.abspath(a.repo_root)
    frame, fe = load_frame(repo_root, script_dir, a.data_csv)

    clusters = sorted(frame["cluster"].dropna().unique().tolist())
    c2f = {c: i for i, c in enumerate(clusters)}
    all_folds = sorted(frame["fold"].unique().tolist())
    print("clusters %s -> folds %s" % (clusters, [c2f[c] for c in clusters]))

    jobs = []
    for mi, mdl in enumerate(a.models):
        for ci, c in enumerate(a.clusters):
            tf = c2f[c]
            n_test = int((frame.fold == tf).sum())
            gpu = (a.gpus[mi % len(a.gpus)] if a.split_by == "model"
                   else a.gpus[ci % len(a.gpus)])
            for vf in [f for f in all_folds if f != tf]:
                jobs.append(dict(model=mdl, cluster=c, test_fold=tf, val_fold=vf,
                                 gpu=gpu, n_test=n_test))

    print("split-by=%s" % a.split_by)
    for g in sorted({j["gpu"] for j in jobs}):
        names = sorted({os.path.basename(j["model"]) for j in jobs if j["gpu"] == g})
        cl = sorted({j["cluster"] for j in jobs if j["gpu"] == g})
        print("   GPU %d -> %s | clusters %s" % (g, ", ".join(names), cl))
    print("%d jobs (%d model(s) x %d clusters x %d inner folds)\n"
          % (len(jobs), len(a.models), len(a.clusters), len(all_folds) - 1))

    # One queue per GPU: a cluster is pinned to a card, jobs within it run serially.
    by_gpu = {}
    for j in jobs:
        by_gpu.setdefault(j["gpu"], []).append(j)

    # Paths must survive being written into a shell script: on Windows the
    # backslashes would be eaten as escapes. shlex.quote plus forward slashes
    # is correct on both platforms.
    q = lambda p: shlex.quote(str(p).replace('\\', '/'))

    procs, t0 = [], time.time()
    for gpu, queue in by_gpu.items():
        script = "\n".join(
            "CUDA_VISIBLE_DEVICES=%d python -u %s --data_csv %s --model %s --seed %d "
            "--gpu_index 0 --max_epochs %d --max_steps %d --patience %d "
            "--worker_test_fold %d --worker_val_fold %d --worker_output_path %s"
            % (gpu, q(a.script), q(a.data_csv), q(j["model"]), a.seed, a.max_epochs,
               a.max_steps, a.patience, j["test_fold"], j["val_fold"],
               q(os.path.join(a.out, "pred__%s__c%d__v%d.csv"
                              % (os.path.basename(j["model"]), j["cluster"], j["val_fold"]))))
            for j in queue)
        sh = os.path.join(a.out, "gpu%d.sh" % gpu)
        open(sh, "w").write("set -e\n" + script + "\n")
        log = open(os.path.join(a.out, "gpu%d.log" % gpu), "w")
        env = dict(os.environ, PYTHONPATH=repo_root + os.pathsep + os.environ.get("PYTHONPATH", ""))
        procs.append((gpu, subprocess.Popen(["bash", sh], stdout=log,
                                            stderr=subprocess.STDOUT, cwd=repo_root, env=env)))
        print("GPU %d: %d jobs launched" % (gpu, len(queue)))

    while any(p.poll() is None for _, p in procs):
        time.sleep(120)
        done = len([f for f in os.listdir(a.out) if f.startswith("pred__")])
        print("[%6.1f min] %d/%d predictions" % ((time.time() - t0) / 60, done, len(jobs)))
    for gpu, p in procs:
        print("GPU %d exit %d" % (gpu, p.returncode))

    # ---- ensemble across inner val folds, score per cluster ----
    rows = []
    for mdl in a.models:
        name = os.path.basename(mdl)
        for c in a.clusters:
            tf = c2f[c]
            truth = frame.loc[frame.fold == tf, "value"].to_numpy(dtype=np.float32)
            preds = []
            for vf in [f for f in all_folds if f != tf]:
                p = os.path.join(a.out, "pred__%s__c%d__v%d.csv" % (name, c, vf))
                if os.path.exists(p):
                    df = pd.read_csv(p).sort_values("row_idx")
                    if len(df) == len(truth):
                        preds.append(df["prediction"].to_numpy(dtype=np.float32))
            if not preds:
                print("no predictions for %s cluster %d" % (name, c)); continue
            ens = np.mean(preds, axis=0)
            rows.append(dict(
                model=name, cluster=c, n_models=len(preds), n_test=len(truth),
                r2=r2_score(truth, ens),
                spearman=float(spearmanr(truth, ens).statistic),
                rmse=float(np.sqrt(mean_squared_error(truth, ens))),
                mae=float(mean_absolute_error(truth, ens)),
                r2_single_mean=float(np.mean([r2_score(truth, p) for p in preds])),
            ))
    res = pd.DataFrame(rows)
    if not res.empty:
        pd.set_option("display.width", 200)
        print("\n" + res.to_string(index=False))
        res.to_csv(os.path.join(a.out, "pampa_metrics.csv"), index=False)
        print("\nReference (paper, cluster-held-out R2): 32M MLM 0.13 | 32M MTR 0.38 | 337M 0.58")
    json.dump({"jobs": len(jobs), "minutes": (time.time() - t0) / 60},
              open(os.path.join(a.out, "run_meta.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
