"""Is there anything CHEMICALLY distinctive about the molecules both models fail on?

Runs on every benchmark where the teacher and the pruned student scored the same
molecules: AmpHGT, CellPPD, PepMSND.

WHY CHARGE AND HYDROPHOBICITY. Two of these three tasks have a known
physicochemical basis, and it is the same one:

    AmpHGT   antimicrobial peptides bind the ANIONIC bacterial membrane, so
             activity needs net POSITIVE charge, and needs hydrophobicity to then
             insert into the bilayer. Cationic-but-not-hydrophobic is the classic
             false lead.
    CellPPD  cell-penetrating peptides are famously arginine-rich (TAT,
             penetratin), so cationicity drives this one too.
    PepMSND  the label's meaning is NOT documented in the repo or in our copy of
             the paper, so its numbers are reported descriptively and not given a
             biological interpretation.

Charge is counted from substructures, not formal charges: guanidine (Arg) and
aliphatic primary amine (Lys) are protonated at pH 7, carboxylic acid (Asp/Glu) is
deprotonated. The patterns must match BOTH protonation states as written, because
the benchmarks disagree -- amp_test.csv is neutral throughout, CellPPD_test.csv
writes 66% of its amines as [NH3+] and 55% of its guanidines as [NH2+].

CAVEAT worth carrying into any writeup: this is a substructure count, not a pKa
calculation, and Crippen logP is poorly calibrated for molecules this large. The
ORDERING across groups is robust; the absolute values are not physical constants.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PAT = {
    "Arg_guanidine": Chem.MolFromSmarts("[NX3][CX3](=[NX2,NX3+])[NX3]"),
    # Both protonation states. AmpHGT writes these SMILES neutral, but CellPPD
    # writes 66% of them protonated -- the neutral-only pattern counted 0.57 Lys
    # per molecule there and silently missed 2.22.
    "Lys_amine": Chem.MolFromSmarts(
        "[$([NX3;H2;!$(NC=O)]),$([NX4;H3;+])][CX4][CX4][CX4][CX4]"),
    "acid_AspGlu": Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]"),
    "aromatic_ring": Chem.MolFromSmarts("a1aaaaa1"),
    "Trp_indole": Chem.MolFromSmarts("c1ccc2[nH]ccc2c1"),
}


def featurize(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return {}
    f = {k: len(m.GetSubstructMatches(p)) for k, p in PAT.items()}
    f["pos_charge"] = f["Arg_guanidine"] + f["Lys_amine"]
    f["neg_charge"] = f["acid_AspGlu"]
    f["net_charge"] = f["pos_charge"] - f["neg_charge"]
    f["MW"] = Descriptors.MolWt(m)
    f["logP"] = Crippen.MolLogP(m)
    f["heavy"] = m.GetNumHeavyAtoms()
    # Size-normalised: raw counts track peptide length, so these compare
    # COMPOSITION rather than "this molecule is bigger".
    f["charge_density"] = f["net_charge"] / max(f["heavy"], 1) * 100
    f["logP_per_heavy"] = f["logP"] / max(f["heavy"], 1) * 100
    return f


COLS = ["net_charge", "charge_density", "logP_per_heavy", "heavy", "MW",
        "aromatic_ring", "Trp_indole"]

BENCH = [("amphgt", "AmpHGT (antimicrobial)", "FP"),
         ("cellppd", "CellPPD (cell-penetrating)", "FN"),
         ("pepmsnd", "PepMSND (label undocumented)", "FN")]

for tag, name, focus in BENCH:
    p = os.path.join(R, "results", "analysis", "error_overlap", "error_overlap_%s.csv" % tag)
    if not os.path.exists(p):
        print("skip %s (run error_overlap.py first)" % tag)
        continue
    d = pd.read_csv(p)
    d = pd.concat([d.reset_index(drop=True),
                   pd.DataFrame([featurize(s) for s in d.smiles])], axis=1)
    d = d[d.MW.notna()]

    print("=" * 96)
    print("%s -- %d molecules" % (name, len(d)))
    print("=" * 96)
    groups = {
        "both wrong": (d.teacher_err == 1) & (d.student_err == 1),
        "teacher only": (d.teacher_err == 1) & (d.student_err == 0),
        "student only": (d.teacher_err == 0) & (d.student_err == 1),
        "both right": (d.teacher_err == 0) & (d.student_err == 0),
    }
    print("%-14s %5s" % ("group", "n") + "".join("%14s" % c[:13] for c in COLS))
    for k, m in groups.items():
        s = d[m]
        if not len(s):
            continue
        print("%-14s %5d" % (k, len(s)) + "".join("%14.2f" % s[c].mean() for c in COLS))

    # The decisive cut. For a benchmark whose shared errors are false positives,
    # compare wrongly-flagged NEGATIVES against correctly-rejected negatives and
    # against the true positives: if the errors look like actives on the driving
    # feature, that is why both models call them.
    side = 0 if focus == "FP" else 1
    other = 1 - side
    sub = d[d.true == side]
    err = sub[(sub.teacher_err == 1) & (sub.student_err == 1)]
    ok = sub[(sub.teacher_err == 0) & (sub.student_err == 0)]
    ref = d[d.true == other]
    if len(err) >= 5:
        print("\n   shared %s (n=%d) vs correctly handled same-class (n=%d), "
              "with the opposite class as the reference the models are confusing "
              "them with (n=%d):" % (focus, len(err), len(ok), len(ref)))
        print("   %-18s %12s %12s %14s %10s"
              % ("feature", "shared " + focus, "correct", "opposite class",
                 "%% toward"))
        for c in COLS:
            a, b, t = err[c].mean(), ok[c].mean(), ref[c].mean()
            frac = 100 * (a - b) / (t - b) if abs(t - b) > 1e-9 else float("nan")
            print("   %-18s %12.2f %12.2f %14.2f %9.0f%%" % (c, a, b, t, frac))
        print("   (%% toward = how far the shared errors sit from the correctly")
        print("    handled molecules toward the class they are confused with;")
        print("    ~100%% means indistinguishable on that feature)")
    else:
        print("\n   only %d shared %s -- too few to characterise" % (len(err), focus))
    print()
    d.to_csv(os.path.join(R, "results", "analysis", "error_chemistry", "error_chemistry_%s.csv" % tag), index=False)

print("wrote results/analysis/error_chemistry/error_chemistry_{amphgt,cellppd,pepmsnd}.csv")
