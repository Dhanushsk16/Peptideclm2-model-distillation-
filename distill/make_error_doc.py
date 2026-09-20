# -*- coding: utf-8 -*-
"""Build docs/shared_error_analysis.docx, the writeup sent to the supervisor.

Answers one question: do the teacher and the student make their false positives
and false negatives on the SAME molecules, and is anything characteristic about
those molecules. Numbers come from error_overlap.py, error_chemistry.py and
bench_latency.py; this script only lays them out.

The doc deliberately says "teacher model" and "student model" with no reference
to how the student was built, and shows results one benchmark at a time.

.docx rather than .md because Google Drive converts it to a Google Doc on upload
with the tables intact.
"""
import os

from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

ACCENT = RGBColor(0x0E, 0x6B, 0x62)
MUTED = RGBColor(0x5C, 0x66, 0x63)

d = Document()
st = d.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(11)
st.paragraph_format.space_after = Pt(8)
st.paragraph_format.line_spacing = 1.15
for name, size in (("Heading 1", 20), ("Heading 2", 14)):
    s = d.styles[name]
    s.font.name = "Calibri"
    s.font.size = Pt(size)
    s.font.bold = True
    s.font.color.rgb = ACCENT
    s.paragraph_format.space_before = Pt(18)
    s.paragraph_format.space_after = Pt(6)


def para(t, italic=False, color=None, size=None, after=None):
    p = d.add_paragraph()
    r = p.add_run(t)
    r.italic = italic
    if color is not None:
        r.font.color.rgb = color
    if size is not None:
        r.font.size = Pt(size)
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def table(header, rows, bold=()):
    t = d.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(9.5)
        if i:
            c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for j, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(10)
            r.bold = j in bold
            if i:
                cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    d.add_paragraph()


def label(t):
    para(t, italic=True, after=4)


d.add_heading("Do Both Models Fail on the Same Molecules?", level=1)
para("Teacher model vs. student model, compared molecule by molecule on the three "
     "benchmarks where both scored an identical test set.", color=MUTED)
para("A molecule counts as an error for a model if it is misclassified in a majority "
     "of that model’s three runs.", italic=True, color=MUTED, size=9.5)

# ---------------------------------------------------------------- cost
d.add_heading("What the student costs to run", level=2)
para("Both timings were measured on CellPPD. Inference: the 300 test molecules, 28,464 "
     "tokens, averaging 135 heavy atoms each, on one RTX 3050. Fine-tuning: the 1,164 "
     "training molecules over five folds, one seed, on one T4. Each row compares the two "
     "models on the same GPU as each other.", color=MUTED, size=10)
table(["", "Teacher", "Student", "Advantage"],
      [["Parameters", "336.7 M", "84.8 M", "4.0×"],
       ["Fine-tuning, one CellPPD seed", "87 min", "31 min", "2.8×"],
       ["Inference latency", "44.1 ms/molecule", "11.1 ms/molecule", "4.0×"],
       ["Throughput", "22.7 molecules/s", "89.8 molecules/s", "4.0×"],
       ["Peak GPU memory", "1,528 MB", "556 MB", "2.8×"]],
      bold=(0, 2))
para("Inference used fp32 at batch 16, each model in its own process, 5 timed passes "
     "after warm-up, with run-to-run spread under 1%. The speed-up tracks the parameter "
     "ratio almost exactly, as expected when the reduction is uniform across the network.")
para("Fine-tuning was timed end to end on the same GPU model, running the same script "
     "over the same five folds. It gains less than 4× because both models train an "
     "identical 1.6 M LoRA parameters — only the forward and backward passes through "
     "the frozen backbone differ — and because per-run overhead does not scale down. "
     "Memory falls by less than 4× for the same reason: the token embeddings and the "
     "activations do not shrink along with the rest.", italic=True, color=MUTED, size=10)

# ---------------------------------------------------------------- AmpHGT
d.add_heading("AmpHGT — antimicrobial activity", level=2)
para("5,118 molecules: 2,666 active, 2,452 inactive.", color=MUTED, size=10)
table(["Model", "TP", "FN", "TN", "FP", "Wrong", "Accuracy"],
      [["Teacher", 2605, 61, 2164, 288, "349 (6.8%)", "0.932"],
       ["Student", 2629, 37, 2052, 400, "437 (8.5%)", "0.915"]])
para("Both models err almost entirely by calling inactive molecules active. The "
     "student is slightly better at finding actives and worse at rejecting inactives "
     "— it shifted toward the positive class rather than degrading evenly.")

label("Error overlap")
table(["", "Student wrong", "Student right"],
      [["Teacher wrong", 233, 116], ["Teacher right", 204, 4565]])
para("233 molecules are wrong for both, against 29.8 expected if the two models erred "
     "independently — 7.8× enrichment, Cohen’s κ = 0.56. "
     "67% of the teacher’s errors are also the student’s.")
table(["Error type", "Teacher", "Student", "Both", "Expected", "Enrichment"],
      [["False positives", 288, 400, 210, "47.0", "4.5×"],
       ["False negatives", 61, 37, 23, "0.8", "27.2×"]])
para("Volume sits in the false positives; concentration sits in the false negatives. "
     "Actives are easy for both models, but when one misses an active the other "
     "usually misses the same one.")

label("What the 210 shared false positives look like")
para("The comparison is between two groups that share the same true label — both "
     "are inactive molecules — so the only difference is that the models fell for "
     "one group and not the other. Actives are shown as the reference.", size=10)
table(["Feature", "210 shared FP", "1,974 correctly rejected", "2,666 actives"],
      [["Heavy atoms", "237", "407", "115"],
       ["   % under 250 atoms", "52%", "6%", "97%"],
       ["Arg (guanidine)", "2.33", "2.82", "1.06"],
       ["Lys (amine)", "2.48", "2.56", "2.37"],
       ["Asp/Glu (acid)", "3.52", "8.86", "0.73"],
       ["Net charge", "+1.29", "−3.48", "+2.69"],
       ["Hydrophobicity / atom", "−5.83", "−5.44", "−1.93"]],
      bold=(0, 4, 5, 6))

label("What each model’s score tracks, across the 2,452 inactive molecules")
table(["Feature", "Teacher", "Student"],
      [["Heavy atoms", "−0.47", "−0.64"],
       ["Net charge", "+0.38", "+0.46"],
       ["Hydrophobicity / atom", "−0.02", "−0.17"]])

para("Size is the strongest driver. 97% of actives are under 250 heavy atoms and only "
     "6% of correctly-rejected inactives are, so “large means inactive” is "
     "nearly free accuracy and both models took it. The 210 errors sit at 237 atoms, "
     "in the zone where that shortcut stops working.")
para("Charge is the second driver, and the mechanism is specific: the shared false "
     "positives are not more cationic — their Arg and Lys counts match the "
     "molecules correctly rejected. They carry less than half the acidic residues "
     "(3.52 vs 8.86), which is what turns net charge positive. The models respond to "
     "the absence of negative charge, not the presence of positive charge.")
para("Hydrophobicity contributes nothing. The shared false positives are marginally "
     "more hydrophilic than the inactives correctly rejected, and neither model’s "
     "score tracks the feature at all.")
para("This matters because antimicrobial activity needs both: positive charge to bind "
     "the anionic bacterial membrane, and a hydrophobic face to then insert into it. "
     "Cationic-but-hydrophilic peptides bind the surface without killing. Both models "
     "learned the binding half of the rule and not the insertion half, so they fail on "
     "the same molecules.")
para("One limit this analysis cannot resolve: whether the models respond to size and "
     "charge as chemistry, or whether both are proxies for “this is a short "
     "peptide rather than a large glycoconjugate,” which is largely what separates "
     "the two classes in this dataset.", italic=True, color=MUTED, size=10)

# ---------------------------------------------------------------- CellPPD
d.add_heading("CellPPD — cell penetration", level=2)
para("291 molecules: 141 penetrating, 150 non-penetrating.", color=MUTED, size=10)
table(["Model", "TP", "FN", "TN", "FP", "Wrong", "Accuracy"],
      [["Teacher", 129, 12, 146, 4, "16 (5.5%)", "0.945"],
       ["Student", 126, 15, 131, 19, "34 (11.7%)", "0.883"]])
para("The student roughly doubles the error rate, and nearly all of the increase is "
     "false positives (4 → 19) — the same shift toward the positive class "
     "seen on AmpHGT.")

label("Error overlap")
table(["", "Student wrong", "Student right"],
      [["Teacher wrong", 8, 8], ["Teacher right", 26, 249]])
para("8 shared errors against 1.87 expected — 4.3× enrichment. The overlap "
     "is real. κ = 0.27 understates it: with only 16 teacher errors in total, "
     "chance-corrected agreement cannot reach a high value regardless of how aligned "
     "the models are.")
table(["Error type", "Teacher", "Student", "Both", "Expected", "Enrichment"],
      [["False positives", 4, 19, 3, "0.51", "5.9×"],
       ["False negatives", 12, 15, 5, "1.28", "3.9×"]])
para("Eight molecules, split three and five, is too few for a descriptor comparison. "
     "The one observation worth recording: the five shared false negatives carry far "
     "fewer arginines than the penetrating peptides the models handle correctly "
     "(0.60 vs 3.84) but more than twice the tryptophan (1.20 vs 0.50 indoles). Net "
     "charge follows, at 3.20 against 6.63. That is the profile of a Trp-driven "
     "cell-penetrating peptide rather than the canonical arginine-rich type, which "
     "both models appear to have learned to the exclusion of the other route. "
     "A hypothesis at n=5, not a result.")
para("A caveat specific to this benchmark: a bag-of-tokens control with no model at "
     "all scores 0.8270, and gradient boosting on Morgan fingerprints reaches 0.8915, "
     "above every model variant. Much of what this benchmark measures is substructure "
     "composition.", italic=True, color=MUTED, size=10)

# ---------------------------------------------------------------- PepMSND
d.add_heading("PepMSND", level=2)
para("640 molecules: 273 positive, 367 negative.", color=MUTED, size=10)
table(["Model", "TP", "FN", "TN", "FP", "Wrong", "Accuracy"],
      [["Teacher", 201, 72, 323, 44, "116 (18.1%)", "0.819"],
       ["Student", 217, 56, 330, 37, "93 (14.5%)", "0.855"]])
para("The student is better on both axes here — more true positives and fewer "
     "false positives. This is also the hardest benchmark for both models.")

label("Error overlap")
table(["", "Student wrong", "Student right"],
      [["Teacher wrong", 72, 44], ["Teacher right", 21, 503]])
para("72 shared errors against 16.9 expected — 4.3× enrichment, κ = "
     "0.63, the strongest agreement of the three benchmarks. 77% of the student’s "
     "errors are also the teacher’s; the student has only 21 errors of its own, "
     "and its advantage comes from fixing 44 of the teacher’s.")
table(["Error type", "Teacher", "Student", "Both", "Expected", "Enrichment"],
      [["False positives", 44, 37, 23, "4.44", "5.2×"],
       ["False negatives", 72, 56, 49, "14.8", "3.3×"]])
para("Unlike the other two benchmarks the shared errors are false-negative dominated: "
     "both models systematically miss the same positive molecules.")
para("No chemical characterisation is offered. The shared errors sit outside the range "
     "spanned by the two classes rather than between them, so the comparison used for "
     "AmpHGT does not apply — and the meaning of this benchmark’s label is "
     "not documented in the repository or in the version of the paper available to us. "
     "Without knowing the property, there is no physicochemical hypothesis to test.")

# ---------------------------------------------------------------- THPep
d.add_heading("THPep — evaluated, but not pairable", level=2)
para("The student was evaluated here with five-fold cross-validation over a stratified "
     "80/20 split — 487 training and 122 test molecules, 35 of them positive — with the "
     "five fold predictions ensembled by mean logit, repeated across three seeds. Every "
     "test molecule is therefore scored by five independently trained models rather than "
     "one.")
para("That result cannot be joined to the teacher’s for this analysis. The repository "
     "ships no THPep split, so ours was generated; the published predictions come from "
     "the authors’ own split and carry no fold column, meaning a single train/test run "
     "against a validation file. Same benchmark, same test-set size and class balance, "
     "but different molecules and a different protocol — so there are no paired samples "
     "to compare errors on.")

# ---------------------------------------------------------------- summary
d.add_heading("Summary", level=2)
para("The student runs 4× faster than the teacher at inference on identical hardware, "
     "in 2.8× less memory.")
para("On all three benchmarks the two models fail together far more often than chance "
     "— 4.3× to 7.8× — so the student inherits the teacher’s "
     "hard cases rather than failing in a new way.")
para("On AmpHGT, the one benchmark with both a known physicochemical basis and enough "
     "shared errors to characterise, those errors have a specific cause: both models "
     "key on molecular size and on the absence of acidic residues, and neither responds "
     "to hydrophobicity — the property that separates a peptide that binds a "
     "bacterial membrane from one that kills.")

for s in d.sections:
    s.left_margin = s.right_margin = Inches(1.0)
    s.top_margin = s.bottom_margin = Inches(0.9)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs",
                   "shared_error_analysis.docx")
d.save(os.path.normpath(OUT))
print("wrote " + os.path.normpath(OUT))
