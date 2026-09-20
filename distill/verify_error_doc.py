"""Regression check: does every number printed in docs/shared_error_analysis.docx
still fall out of the data?

make_error_doc.py carries its tables as string literals, because several of them
were derived interactively rather than by a script. That is fine for a writeup
but it means a change to the underlying results -- a rerun, a fixed SMARTS
pattern, a moved directory -- can silently leave the doc stating numbers nothing
produces any more. This re-derives all of them and fails loudly on a mismatch.

Run after error_overlap.py and error_chemistry.py:

    python distill/verify_error_doc.py

Exits 1 if anything drifted.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
A = os.path.join(R, "results", "analysis")

# (TP, FN, TN, FP) exactly as tabulated in the doc, per benchmark per model.
CONFUSION = {
    "amphgt": {"teacher": (2605, 61, 2164, 288), "student": (2629, 37, 2052, 400)},
    "cellppd": {"teacher": (129, 12, 146, 4), "student": (126, 15, 131, 19)},
    "pepmsnd": {"teacher": (201, 72, 323, 44), "student": (217, 56, 330, 37)},
}

# feature -> (shared FP, correctly rejected, true actives)
AMPHGT_DESCRIPTORS = {
    "heavy": (237, 407, 115),
    "Arg_guanidine": (2.33, 2.82, 1.06),
    "Lys_amine": (2.48, 2.56, 2.37),
    "acid_AspGlu": (3.52, 8.86, 0.73),
    "net_charge": (1.29, -3.48, 2.69),
    "logP_per_heavy": (-5.83, -5.44, -1.93),
}

# feature -> (teacher, student) correlation with model score across the inactives
AMPHGT_CORRELATIONS = {
    "heavy": (-0.47, -0.64),
    "net_charge": (0.38, 0.46),
    "logP_per_heavy": (-0.02, -0.17),
}

failures = []


def check(label, got, expected, tol=0.015):
    """Relative tolerance, with an absolute floor so values near zero (the
    hydrophobicity correlation at -0.02) are not held to an impossible bar."""
    ok = abs(got - expected) <= tol * max(abs(expected), 1e-9) or abs(got - expected) < 0.006
    if not ok:
        failures.append("%s: doc says %s, data gives %s" % (label, expected, round(got, 4)))
    print("  %-38s %-10s %-10s %s"
          % (label, expected, round(got, 4), "ok" if ok else "MISMATCH"))


def groups(bench):
    """Shared false positives, correctly-rejected negatives, and the positives
    they are being confused with."""
    d = pd.read_csv(os.path.join(A, "error_chemistry", "error_chemistry_%s.csv" % bench))
    neg = d[d.true == 0]
    return (neg[(neg.teacher_err == 1) & (neg.student_err == 1)],
            neg[(neg.teacher_err == 0) & (neg.student_err == 0)],
            d[d.true == 1], neg)


print("confusion matrices")
for bench, models in CONFUSION.items():
    d = pd.read_csv(os.path.join(A, "error_overlap", "error_overlap_%s.csv" % bench))
    for who, expected in models.items():
        e = d[who + "_err"]
        got = (int(((d.true == 1) & (e == 0)).sum()), int(((d.true == 1) & (e == 1)).sum()),
               int(((d.true == 0) & (e == 0)).sum()), int(((d.true == 0) & (e == 1)).sum()))
        for name, g, x in zip(("TP", "FN", "TN", "FP"), got, expected):
            check("%s %s %s" % (bench, who, name), g, x, tol=0)

fp, ok, pos, neg = groups("amphgt")

print("\nAmpHGT descriptor table")
for col, (a, b, c) in AMPHGT_DESCRIPTORS.items():
    check(col + " shared FP", fp[col].mean(), a)
    check(col + " correctly rejected", ok[col].mean(), b)
    check(col + " actives", pos[col].mean(), c)
check("% under 250 shared FP", 100 * (fp.heavy < 250).mean(), 52, tol=0.02)
check("% under 250 correctly rejected", 100 * (ok.heavy < 250).mean(), 6, tol=0.2)
check("% under 250 actives", 100 * (pos.heavy < 250).mean(), 97, tol=0.02)

print("\nAmpHGT score correlations, across the %d inactives" % len(neg))
for col, (t, s) in AMPHGT_CORRELATIONS.items():
    check(col + " teacher", neg.teacher_score.corr(neg[col]), t, tol=0.03)
    check(col + " student", neg.student_score.corr(neg[col]), s, tol=0.03)

print("\nCellPPD shared false negatives")
c = pd.read_csv(os.path.join(A, "error_chemistry", "error_chemistry_cellppd.csv"))
fn = c[(c.true == 1) & (c.teacher_err == 1) & (c.student_err == 1)]
right = c[(c.true == 1) & (c.teacher_err == 0) & (c.student_err == 0)]
check("n shared FN", len(fn), 5, tol=0)
for col, (a, b) in {"Arg_guanidine": (0.60, 3.84), "Trp_indole": (1.20, 0.50),
                    "net_charge": (3.20, 6.63)}.items():
    check(col + " shared FN", fn[col].mean(), a)
    check(col + " correctly handled", right[col].mean(), b)

print()
if failures:
    print("%d MISMATCH(ES) -- docs/shared_error_analysis.docx is stale:" % len(failures))
    for f in failures:
        print("   " + f)
    sys.exit(1)
print("all checks passed -- every number in the doc reproduces from the data")
