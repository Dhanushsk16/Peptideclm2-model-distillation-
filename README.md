# PeptideCLM-2 Model Distillation

Distilling the 337M-parameter **PeptideCLM-2** chemical language model into its
32M-parameter counterpart, and testing whether the distilled student is actually
better at the tasks the models exist for.

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

That gap is what makes distillation interesting here. The 337M model is accurate
but expensive; the 32M model runs anywhere but is markedly weaker. **Can the small
model be taught to behave like the large one?**

## Approach

A 32M student, warm-started from the authors' released `peptideclm-2-mlm-small`,
is trained against three signals at once:

```
L  =  0.30 · L_KD  +  0.30 · L_MTR  +  0.40 · L_SPKD
```

| term | signal | source |
|---|---|---|
| `L_KD` | teacher's token distribution at masked positions | 337M teacher |
| `L_MTR` | 99 RDKit physicochemical descriptors | ground truth |
| `L_SPKD` | pairwise similarity structure within a batch | 337M teacher |

Weights are calibrated from measured loss magnitudes rather than inherited from
the paper — see [`REPORT.md`](REPORT.md) for why that matters.

**A control arm makes the result attributable.** A second student is trained on
identical data, for identical steps, from an identical starting point, with the
teacher removed:

```
L  =  0.70 · L_MLM  +  0.30 · L_MTR
```

Any difference between the two arms is caused by the teacher and nothing else.

## What is in here

```
distill/            training, loss, data and evaluation code
kaggelscripts/      end-to-end notebooks (Kaggle T4 x2)
results/            metrics, predictions and training curves
REPORT.md           full write-up of everything done
```

Model weights, the pretraining corpus and the teacher cache are **not** committed —
they are large and mostly not ours. Every one is reproducible from the notebooks,
and the sources are listed in the report.

## Status

The student demonstrably learns the teacher's behaviour: KL divergence to the
teacher drops by 1.8x to 4.3x across four held-out benchmark sets, while the
control drifts away on all of them.

Whether that translates into better downstream task performance is the open
question, and the current answer is **not yet** — see the report for the
full-finetune PAMPA results and the caveats attached to them.

## Reproducing

Notebooks run in order on Kaggle (2x T4, internet enabled). Google Drive is used
for artifact storage via `rclone`.

| notebook | what it does | runtime |
|---|---|---|
| `kaggle_cellppd_reproduction.ipynb` | reproduce the paper's CellPPD benchmark | ~2.5 h |
| `kaggle_build_pretrain_subset.ipynb` | sample a 2M-molecule training corpus | ~25 min |
| `kaggle_build_teacher_cache.ipynb` | precompute teacher targets | ~10 h |
| `kaggle_distill_train.ipynb` | train both arms | ~6 h |
| `kaggle_pampa_eval.ipynb` | evaluate on PAMPA permeability | ~5 h |

## Credits

Original models, data and benchmarks by Feller, Secor, Swanson, Wilke and Deibler.
Paper: *Scaling SMILES-Based Chemical Language Models for Therapeutic Peptide
Engineering*, bioRxiv 2026.01.06.697994.
Their code: <https://github.com/AaronFeller/PeptideCLM-2> ·
weights: <https://huggingface.co/aaronfeller>

Distillation work in this repository is independent and not affiliated with the
original authors.
