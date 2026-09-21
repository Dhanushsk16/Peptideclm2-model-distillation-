# Making PeptideCLM-2 Cheaper

Two attempts at shrinking the 337M-parameter **PeptideCLM-2** chemical language
model while keeping what it can do. The first — knowledge distillation into a 32M
student — did not work. The second — deleting three quarters of the blocks and
training nothing — did.

---

## The question

[PeptideCLM-2](https://doi.org/10.64898/2026.01.06.697994) (Feller et al., Novo
Nordisk + UT Austin) is a suite of SMILES-based transformer encoders for
therapeutic peptides, released at three scales — 32M, 114M and 337M — trained with
three objectives (masked language modelling, multi-task regression onto RDKit
descriptors, and a hybrid).

Their central result is a **scaling transition**: at 32M parameters, explicit
physicochemical supervision is essential (R² 0.38 vs 0.13 on membrane
permeability), while at 337M a purely self-supervised model recovers the same
ability on its own (both reach R² 0.58).

The 337M model is accurate but expensive; the 32M model runs anywhere but is
markedly weaker. **Can the gap be closed without paying for 337M parameters?**

## Two approaches

| | distillation | training-free compression |
|---|---|---|
| method | train a 32M student against the teacher | drop 24 of 32 blocks, keep the rest bit-identical |
| cost | ~16 GPU-hours per arm | none |
| size | 336.7M → 31.9M (10.5×) | 336.7M → 84.8M (4.0×) |
| downstream | **behind its own starting point** | **+0.077 / +0.037 / −0.047 / −0.102 MCC** |

### 1. Distillation — the negative result

A 32M student, warm-started from `peptideclm-2-mlm-small`, trained against the
teacher's token distribution, RDKit descriptors and batch similarity structure,
with a **control arm** trained identically minus the teacher so any difference is
attributable. Later replaced by pure Hinton KD against a live teacher (T=4,
α=0.95, full vocabulary, fresh mask each epoch).

The student demonstrably moves toward the teacher: KL divergence drops 1.8×–4.3×
across four held-out benchmarks, 93.8% token agreement, while the control drifts
away on all of them.

It gained nothing from doing so. On PAMPA permeability at two training budgets,
and on CellPPD with LoRA, the distilled arms sit **below the undistilled
warm-start they began from** (CellPPD MCC 0.827 and 0.825 against 0.847).
Matching a larger model's distributions and geometry is not sufficient to
transfer its downstream capability.

### 2. Training-free compression — what worked

Block influence (`BI = 1 − E[cos(x_in, x_out)]`) over the 32 blocks is almost
entirely concentrated in **block 0 (0.762) and block 31 (0.604)**; blocks 8–15
score ~0.002, three hundred times lower. Keeping 8 blocks — **0, 1, 2, 3, 5, 6,
10, 16** — and renumbering them into a valid checkpoint gives a 84.8M model with
no training at all.

MCC across three seeds, running the authors' own scripts unmodified:

| benchmark | teacher 336.7M | student 84.8M | Δ |
|---|---|---|---|
| PepMSND | 0.6180 ± 0.004 | **0.6548 ± 0.025** | +0.037 |
| THPep | 0.7557 ± 0.019 | **0.8329 ± 0.013** | +0.077 |
| AmpHGT | **0.8845 ± 0.013** | 0.8374 ± 0.012 | −0.047 |
| CellPPD | **0.8750 ± 0.004** | 0.7736 ± 0.007 | −0.102 |

Two caveats carried in the data itself: THPep compares their single split against
our 5-fold CV, and on CellPPD a bag-of-tokens control with no model scores 0.827,
so that benchmark ranks largely by substructure composition.

Measured cost: **4.0× faster inference** (44.1 → 11.1 ms/molecule), 2.8× less
memory, 2.8× faster fine-tuning.

## Do the two models fail the same way?

Yes — 4.3× to 7.8× more overlap in their errors than independent models would
produce (κ 0.27–0.63). On AmpHGT the 210 shared false positives have a specific
cause: both models key on molecular size and on the *absence* of acidic residues,
and neither responds to hydrophobicity — the property separating a peptide that
binds a bacterial membrane from one that kills.

Full writeup: [`docs/shared_error_analysis.docx`](docs/shared_error_analysis.docx).

## What is in here

```
distill/                 all code — training, probes, export, analysis
kaggelscripts/kd/        distillation notebooks
kaggelscripts/pruning/   compression notebooks
kaggelscripts/baselines/ reproduction of the authors' published results
results/kd/              distillation runs, PAMPA, CellPPD LoRA
results/pruning/         depth and slice probes, block influence, benchmarks
results/analysis/        error overlap and chemistry, latency, summary tables
results/baselines/       our reproduction of their CellPPD result
docs/                    the writeup sent to the supervisor
REPORT.md                full work log, including what turned out to be wrong
```

Model weights, the pretraining corpus and the teacher cache are **not** committed —
they are large and mostly not ours. Every one is reproducible from the notebooks,
and the sources are listed in the report.

## Reproducing

Notebooks run on Kaggle (2× T4, internet enabled). Google Drive is used for
artifact storage via `rclone`.

**Compression** — `kaggelscripts/pruning/`

| notebook | what it does | runtime |
|---|---|---|
| `kaggle_bi_probe.ipynb` | block influence over all 32 blocks | ~20 min |
| `kaggle_slice_probe.ipynb` | prefix/mid/suffix 16-block slices | ~2 h |
| `kaggle_bi_confirm.ipynb` | BI-selected vs naive truncation | ~3 h |
| `kaggle_bisel8_sweep_nb1.ipynb`, `..._nb2.ipynb` | AmpHGT + PepMSND, 3 seeds | ~10 h each |
| `kaggle_bisel8_cls.ipynb` | THPep + CellPPD, 3 seeds | ~4 h |

**Distillation** — `kaggelscripts/kd/`

| notebook | what it does | runtime |
|---|---|---|
| `kaggle_build_pretrain_subset.ipynb` | sample a 2M-molecule corpus | ~25 min |
| `kaggle_build_teacher_cache.ipynb` | precompute teacher targets | ~10 h |
| `kaggle_distill_train.ipynb` | train both arms | ~6 h |
| `kaggle_kd_live_train.ipynb` | live-teacher Hinton KD | ~11 h |
| `kaggle_pampa_eval.ipynb` | PAMPA permeability | ~5 h |
| `kaggle_cellppd_lora.ipynb` | CellPPD with LoRA | ~3 h |

**Analysis** — local, CPU only

```
python distill/error_overlap.py      # who fails on what
python distill/error_chemistry.py    # what those molecules look like
python distill/verify_error_doc.py   # re-derive every number in the doc
```

## Credits

Original models, data and benchmarks by Feller, Secor, Swanson, Wilke and Deibler.
Paper: *Scaling SMILES-Based Chemical Language Models for Therapeutic Peptide
Engineering*, bioRxiv 2026.01.06.697994.
Their code: <https://github.com/AaronFeller/PeptideCLM-2> ·
weights: <https://huggingface.co/aaronfeller>

Work in this repository is independent and not affiliated with the original
authors.
