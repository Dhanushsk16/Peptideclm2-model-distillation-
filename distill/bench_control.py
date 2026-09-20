"""Can these benchmarks tell two encoders apart at all?

Before spending GPU hours comparing compressed models, check that the benchmarks
can resolve a difference. The control is a model with NO transformer: count how
many times each of the 405 vocabulary tokens appears in the SMILES string and fit
a linear model on those counts.

If counting tokens scores as well as the 337M encoder, the benchmark is measuring
composition, not learned representation, and it cannot distinguish a compressed
backbone from an uncompressed one. Measured on CellPPD already: bag-of-tokens
0.8270 vs the full encoder's 0.8278. That benchmark is blind.

PepMSND gets a second control, because its pipeline feeds ~140 RDKit descriptors
and Species/Environment metadata alongside the encoder. If the descriptors alone
carry the signal, the encoder is a passenger there regardless of its size.

Also creates the THPep train/test split, which their repo does not ship -- their
classification script looks for THPep_train.csv / THPep_test.csv and crashes
without them. The split is ours, so THPep numbers are comparable between our arms
but NOT to the paper's.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import matthews_corrcoef, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(R, "their_repo", "data")

# What the full 337M encoder scores, for the comparison that matters. CellPPD is
# measured (linear probe on its pooled output, same protocol as here). The rest
# are filled in as they are measured.
ENCODER_REF = {"CellPPD": 0.8278}


def counts(tok, smiles, V=405, max_length=512):
    X = np.zeros((len(smiles), V), dtype=np.float32)
    for i, s in enumerate(smiles):
        for t in tok(str(s), add_special_tokens=False, truncation=True,
                     max_length=max_length)["input_ids"]:
            X[i, t] += 1
    return X


def clf(Xtr, ytr, Xte, yte):
    best, bs = None, -2
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    for C in (0.01, 0.1, 1.0):
        s = [matthews_corrcoef(
                ytr[va], make_pipeline(StandardScaler(),
                                       LogisticRegression(C=C, max_iter=1000)
                                       ).fit(Xtr[tr], ytr[tr]).predict(Xtr[va]))
             for tr, va in skf.split(Xtr, ytr)]
        if np.mean(s) > bs:
            best, bs = C, np.mean(s)
    p = make_pipeline(StandardScaler(), LogisticRegression(C=best, max_iter=1000))
    p.fit(Xtr, ytr)
    return dict(mcc=matthews_corrcoef(yte, p.predict(Xte)),
                auc=roc_auc_score(yte, p.predict_proba(Xte)[:, 1]), n_te=len(yte))


def make_thpep_split(seed=101, test_frac=0.2):
    """Stratified split written where their script expects it."""
    out_tr, out_te = os.path.join(D, "THPep_train.csv"), os.path.join(D, "THPep_test.csv")
    src = pd.read_csv(os.path.join(D, "THPep_main90_smiles_classes.csv"))
    df = src.rename(columns={"class": "label"})[["smiles", "label"]].dropna()
    tr, te = train_test_split(df, test_size=test_frac, stratify=df.label,
                              random_state=seed)
    tr.to_csv(out_tr, index=False)
    te.to_csv(out_te, index=False)
    print("THPep split written: train %d (%d pos) | test %d (%d pos)  [OURS, not theirs]"
          % (len(tr), tr.label.sum(), len(te), te.label.sum()))
    return tr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", default=os.path.join(R, "results", "analysis", "bench_control.json"))
    a = ap.parse_args()

    tk = a.tokenizer or glob.glob(
        R + "/models/models--aaronfeller--peptideclm-2-mlm-small/snapshots/*")[0]
    tok = AutoTokenizer.from_pretrained(tk, trust_remote_code=True)
    res = {}

    # ---------------------------------------------------------------- THPep
    thp_tr, thp_te = make_thpep_split()

    # ------------------------------------------------- classification benchmarks
    sets = {
        "CellPPD": (pd.read_csv(D + "/CellPPD_train.csv"), pd.read_csv(D + "/CellPPD_test.csv")),
        "AmpHGT": (pd.read_csv(D + "/amp_train.csv"), pd.read_csv(D + "/amp_test.csv")),
        "THPep": (thp_tr, thp_te),
    }
    print("\n%-10s %8s %8s %8s %10s" % ("benchmark", "MCC", "AUC", "n_test", "encoder"))
    for name, (tr, te) in sets.items():
        r = clf(counts(tok, tr.smiles), tr.label.to_numpy().astype(int),
                counts(tok, te.smiles), te.label.to_numpy().astype(int))
        res[name] = r
        ref = ENCODER_REF.get(name)
        print("%-10s %8.4f %8.4f %8d %10s"
              % (name, r["mcc"], r["auc"], r["n_te"],
                 "%.4f" % ref if ref else "not yet"))

    # ---------------------------------------------------------------- PepMSND
    # 10 pre-made folds. Two controls: token counts, and the ~140 RDKit
    # descriptors their own pipeline already feeds the model.
    META = ["ID", "PMID", "SMILES", "label", "Length", "SE-3", "Species", "Environment"]
    tok_m, desc_m = [], []
    for k in range(1, 11):
        tr = pd.read_csv(D + "/PepMSND_data/X_train%d.csv" % k)
        te = pd.read_csv(D + "/PepMSND_data/X_test%d.csv" % k)
        ytr = (tr.label.to_numpy() >= 0.5).astype(int)
        yte = (te.label.to_numpy() >= 0.5).astype(int)
        tok_m.append(clf(counts(tok, tr.SMILES), ytr, counts(tok, te.SMILES), yte)["mcc"])
        dcols = [c for c in tr.columns if c not in META]
        dtr = tr[dcols].to_numpy(np.float32); dte = te[dcols].to_numpy(np.float32)
        dtr = np.nan_to_num(dtr); dte = np.nan_to_num(dte)
        desc_m.append(clf(dtr, ytr, dte, yte)["mcc"])
    res["PepMSND"] = dict(mcc=float(np.mean(tok_m)), mcc_std=float(np.std(tok_m)),
                          mcc_desc=float(np.mean(desc_m)), mcc_desc_std=float(np.std(desc_m)),
                          n_te=len(yte))
    print("%-10s %8.4f %8s %8d %10s   (descriptors-only %.4f +- %.4f)"
          % ("PepMSND", res["PepMSND"]["mcc"], "-", res["PepMSND"]["n_te"], "not yet",
             res["PepMSND"]["mcc_desc"], res["PepMSND"]["mcc_desc_std"]))

    # ---------------------------------------------------------------- PAMPA
    # Regression, and their protocol holds out a whole cluster. Ridge on counts,
    # R2 on the held-out cluster -- the same quantity the finetuning reports.
    p = pd.read_csv(D + "/PAMPA_clusters.csv")
    Xall = counts(tok, p.SMILES)
    pam = {}
    for c in (1, 6):
        m = p.cluster == c
        r = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xall[~m.values], p.PAMPA[~m].values)
        pam[c] = float(r2_score(p.PAMPA[m].values, r.predict(Xall[m.values])))
    res["PAMPA"] = dict(r2_cluster1=pam[1], r2_cluster6=pam[6])
    print("%-10s  R2 cluster1 %+.4f  cluster6 %+.4f   (encoder run2: 0.066 / 0.320)"
          % ("PAMPA", pam[1], pam[6]))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("\nwrote " + a.out)
    print("\nRead this as: any benchmark where the control matches the encoder cannot")
    print("measure compression, and should be dropped from the comparison.")


if __name__ == "__main__":
    main()
