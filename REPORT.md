# Distilling PeptideCLM-2: work log and results

A full account of what was done, what was measured, and what turned out to be
wrong. Numbers here are the ones that survived re-checking; where an earlier
measurement was retracted, that is stated rather than quietly dropped.

---

## 1. Background

[PeptideCLM-2](https://doi.org/10.64898/2026.01.06.697994) is a suite of nine
SMILES-based transformer encoders for therapeutic peptides — three parameter
scales (32M / 114M / 337M) crossed with three pretraining objectives (MLM, MTR,
Hybrid). Inputs are raw SMILES strings, tokenized with a custom 405-token k-mer
vocabulary that compresses peptide SMILES ~60% versus atom-level encoding.

Architecture: BERT-style encoder, RoPE, SwiGLU, pre-layer-norm, head dimension
fixed at 64.

Their headline finding is a **scaling transition**. At 32M, explicit
physicochemical supervision (MTR) is decisive on membrane permeability — R² 0.38
versus 0.13 for MLM alone. At 337M, the purely self-supervised model catches up
entirely; both reach R² ≈ 0.58.

The distillation question follows directly: the capability exists only at the
large scale, and the paper's own evidence says small models cannot reach it by
pretraining alone.

### Verified against the released checkpoints

| | released config | paper §4.3.1 |
|---|---|---|
| large | 32 blocks, d=1024, ffn 2048, 16 heads → **336.7M** ✓ | "24 layers, 1024 hidden" |
| small | 14 blocks, d=512, ffn 768, 8 heads → **31.7M** ✓ | "6 layers, 384 hidden, 6 heads" |

Parameter totals match the paper; the layer/width breakdown does not. The
checkpoints are authoritative.

The released MTR checkpoints contain **no descriptor head** — only `embed`,
`transformer.blocks.*` and a 405-dim `sequence_head`. So the teacher exposes token
logits, hidden states and a mean-pooled embedding, not descriptor predictions.

---

## 2. Reproducing the paper first

Before building anything, the published CellPPD classification benchmark was
reproduced using the authors' own script, unmodified.

**Result** (test MCC, ensemble of 5 CV folds, 3 seeds):

| seed | ours | theirs | Δ |
|---|---|---|---|
| 101 | 0.8953 | 0.8760 | +0.019 |
| 202 | 0.8749 | 0.8780 | −0.003 |
| 303 | 0.8946 | 0.8711 | +0.024 |
| **mean** | **0.8883** | **0.8750** | **+0.013** |

Reproduced. Two things worth recording:

**The aggregation is not exactly recoverable.** Seven plausible aggregations were
tested against their published `cellppd_all.csv`; none reproduces it exactly. The
closest (ensemble by mean logit, threshold 0) is 0.009 off on average and 0.020 at
worst, measured on *their own* shipped predictions. So ±0.02 MCC is the noise floor
for any comparison against their published numbers.

**Fold collapse is a real failure mode.** In the authors' own `mlm-large` seed 101,
two of five folds finished at MCC 0.000 — the head never separated the classes.
Their ensemble still scored 0.876 because averaging logits over the three healthy
folds carried it.

### Finetuning strategy differs by benchmark

| script | benchmarks | trunk |
|---|---|---|
| `02_classification.../classification_finetuning_v2.py` | CellPPD, THPep, AmpHGT | **frozen**, LoRA only |
| `01_regression.../finetune_ensemble.py` | PAMPA / CycPeptMPDB | **fully trainable** |

The classification path trains ~3.15M of 336.7M parameters (**0.93%**): LoRA
(r=16, α=32) on `qkv_proj` plus a 2-layer head. Three places where the released
code contradicts the paper's §4.4.1: the paper describes full finetuning, lr 1e-5,
and "two hidden layers with GeLU"; the code does LoRA, lr 3e-4, and one hidden
layer with SiLU. The published numbers came from the code.

---

## 3. Building the training corpus

The full pretraining corpus is 34.35 GB / 117.9M molecules on HuggingFace. Only a
subset is needed: for a 32M student, ~20 tokens/parameter is ≈640M tokens, which
at ~160 tokens/molecule is ~4M molecules for a single compute-optimal pass.

**A 2,023,640-molecule subset was built**, fetching **1.14 GB instead of 34 GB**.

### The filenames are swapped

Verified against the `source` column and the row counts in the paper:

| file | actual contents | rows |
|---|---|---|
| `train/peptides.parquet` | **PubChem small molecules** | 108,116,047 |
| `train/small_molecules.parquet` | **ESMAtlas peptides** | 9,538,596 |
| `train/lipids.parquet` | LMSD lipids ✓ | 48,654 |

Sampling on the filenames would have inverted the corpus entirely. Every reference
in the builder goes through a verified mapping, and the `source` column is
re-asserted on the data actually pulled.

### Sampling method

Parquet's unit of access is the row group, so drawing 1M uniformly random rows out
of 108M would touch nearly every group — i.e. download the whole file. Instead the
builder reads a **stride of row groups** across each file and subsamples within
them.

This is safe because the files are effectively pre-shuffled: KS tests between the
first and last row group gave D = 0.012–0.018, p = 0.39–0.84 on MolWt, MolLogP and
TPSA — statistically indistinguishable.

### Composition and filters

| source | rows | share |
|---|---|---|
| PubChem small molecules | 1,006,000 | 49.7% |
| ESMAtlas peptides | 969,000 | 47.9% |
| LMSD lipids | 48,640 | 2.4% |

Filters: ≤512 k-mer tokens (drops ~4.5% of peptides; attention is quadratic and
they reach 592), and **cross-source deduplication** — deduping per source is not
enough, because LMSD lipids are drug-like and also appear in PubChem. The pooled
frame is deduped with the rarest source winning.

Final: **320,232,341 tokens**, mean 158.2/molecule, zero duplicates, zero NaN.

The 99 RDKit descriptors ship **pre-normalized** (z-scored globally — stored MolWt
is −0.43 for PubChem, +1.85 for peptides). They must not be normalized again or
recomputed with RDKit; both would rescale the MTR targets away from what the
teacher was trained against. They are *not* winsorized (range −13.25 to +583), so
clipping at ±10 is applied at load time — 0.007% of cells.

---

## 4. Caching the teacher

The teacher is frozen, so its outputs can be precomputed once. ~10 hours on 2× T4.

| cached | purpose | format |
|---|---|---|
| top-16 log-probs at masked positions | `L_KD` | fp16 |
| vocab indices | `L_KD` | int16 |
| full-vocab logsumexp | exact renormalisation | fp32 |
| mean-pool (unmasked forward) | `L_SPKD` | fp16 |
| masked positions | alignment | int16, CSR |

**k = 16 was chosen by measurement, not convention.** Probability mass captured at
masked positions on the real teacher:

| k | mean | p05 |
|---|---|---|
| 8 | 0.925 | **0.633** |
| **16** | **0.962** | **0.803** |
| 32 | 0.984 | 0.917 |

The mean is the wrong statistic — on the hardest 5% of positions top-8 discards a
third of the distribution, precisely where the teacher is uncertain and the soft
target carries the most information. k=16 costs +2.6 GB.

**Log-probs are stored, not raw logits.** Raw logits in fp16 have ~0.016 resolution
at magnitude ~20, and `exp()` amplifies that into `Σp = 1.0075` — a *negative*
residual, which would produce NaN in the KL term. Storing `logit − logsumexp`
puts values near 0 where fp16 is fine-grained: max sum 1.00006, max error 1.5e-4.
Same bytes, 50× more accurate.

**Two disjoint masks per molecule** (A and B, 25% each, non-overlapping), verified
to achieve the full budget at every sequence length with zero overlap. Epoch 0 uses
A, epoch 1 uses B — deterministic alternation, because with 2 epochs random choice
leaves half the molecules seeing one mask twice.

Masking replicates their `pretraining.py` exactly: per-sequence 25% budget, spans
~𝒩(3.5, 1.0), overlap rejection, applied before `[CLS]`/`[SEP]`. One deliberate
deviation — their loop has no escape and can spin forever when no non-overlapping
span fits; a guard was added.

---

## 5. The distillation objective

```
L  =  λ_KD · L_KD  +  λ_MTR · L_MTR  +  λ_SPKD · L_SPKD
```

**`L_KD`** — KL divergence over 17 buckets: the teacher's top-16 plus a lumped
residual. The residual keeps the discarded ~4% tail honest instead of fitting the
student to a distribution that sums to 0.96. Temperature is kept at T = 1, because
only at T=1 are the cached probabilities exact; any T ≠ 1 needs renormalising over
all 405 logits and would redistribute mass into precisely the tail that was
discarded.

**`L_MTR`** — MSE against the 99 descriptors. Not distillation: these are ground
truth and the teacher is not involved.

**`L_SPKD`** — similarity-preserving KD. Row-normalised Gram matrices are compared:

```
G = rownorm(Z Zᵀ),     L_SPKD = ‖G_student − G_teacher‖²_F / B²
```

This is necessary because **independently trained models occupy unrelated bases**.
Measured: cosine similarity between the MLM-large and MTR-large embeddings of the
*same molecule* is **0.0044** — essentially orthogonal. Gram matrices are invariant
to rotation, `(ZR)(ZR)ᵀ = ZZᵀ`, and collapse the width mismatch (student 512,
teacher 1024) to B×B, so no projection head is needed.

### Why not the paper's 0.6 / 0.4 split

Their split assumed an MLM loss starting near 6 nats (training from scratch). A
warm-started student starts at **KL 0.078** with 91.8% top-1 agreement with the
teacher. Inheriting 0.6/0.4 would let MTR dominate KD roughly 8:1 and make SPKD
invisible.

Instead each λ is set so the term contributes its intended share:
`λ_k = share_k / L_k(measured)` → KD 30%, MTR 30%, SPKD 40%.

SPKD carries the largest share deliberately: the warm-started student already
matches the teacher as a *token predictor*, so KD has limited headroom, but it does
not match the teacher's representation geometry — which is what the paper's
downstream gap is about.

**Calibration must happen after warmup.** Measured on this pipeline: at step 20,
`L_MTR` = 4.30 (untrained head) → λ_MTR = 0.07; by step 1100, `L_MTR` = 0.27 →
λ_MTR = 1.11. A 16× error that would have silently under-weighted MTR for the whole
run, with no crash to reveal it. Default is now step 2500, window 200.

---

## 6. Training

Two arms, 39,678 steps (2 epochs), ~6 hours on 2× T4.

| | treatment | control |
|---|---|---|
| loss | `0.30 L_KD + 0.30 L_MTR + 0.40 L_SPKD` | `0.70 L_MLM + 0.30 L_MTR` |
| teacher | yes | **no** |
| init, data, schedule, steps | identical | identical |

`L_MLM` is standard cross-entropy against the true token at the same masked
positions — identical task, one-hot supervision instead of a distribution.

Hyperparameters: student LR 1e-4 (a third of their 3e-4 pretraining peak, since we
start from trained weights), head LR 3e-4, AdamW β=(0.9, 0.98), wd 0.01, grad clip
1.0, 2000-step warmup then cosine to 10%, token-budget batching at 16,384 tokens.

**MTR saturates almost immediately** — R² ≈ 0.98 by step 6,800. The warm-started
encoder already encodes the descriptors, so the term contributes little gradient
for the remaining 83% of training. The paper's claim that 32M models need MTR
applies to training *from scratch*; bolted onto a pretrained encoder it is solved
on arrival.

---

## 7. Results

### Teacher agreement — training corpus

| model | KL (mask A) | top-1 agree | KL (mask B) | agree |
|---|---|---|---|---|
| warm-start | 0.0756 | 0.914 | 0.0759 | 0.912 |
| **treatment** | **0.0187** | **0.955** | **0.0190** | **0.957** |
| control | 1.0422 | 0.863 | 1.0579 | 0.862 |

Treatment is **4× closer** to the teacher than its own starting point; control
drifted **14× further away**. Identical on both masks, so it generalises past the
mask it trained on.

### Teacher agreement — held-out benchmarks

Teacher run **live** at full precision on molecules outside the distillation
corpus. Full 405-way KL, no top-k truncation.

| benchmark | warm-start | treatment | control |
|---|---|---|---|
| PAMPA (permeability) | 0.288 | **0.160** (1.8×) | 0.388 |
| CellPPD (cell penetration) | 0.395 | **0.250** (1.6×) | 0.967 |
| AmpHGT (antimicrobial) | 0.219 | **0.106** (2.1×) | 1.189 |
| THPep (tumour homing) | 0.155 | **0.036** (4.3×) | 1.154 |

Top-1 agreement rises on every set (e.g. THPep 0.898 → 0.946). Nothing here can be
memorisation — these molecules were never trained on and the teacher was not read
from cache.

### Representation geometry

1,024 molecules, 523,776 pairs. Each model embedded in a **separate process** (see
§8).

| model | SPKD | raw \|Δcos\| | centered | Spearman | cross-cos |
|---|---|---|---|---|---|
| teacher | — | — | — | — | 0.5139 |
| warm-start | 2.20e-05 | 0.0558 | 0.0533 | 0.9021 | 0.4941 |
| **treatment** | **1.10e-05** | **0.0440** | **0.0442** | **0.9424** | **0.5082** |
| control | 1.89e-04 | 0.1889 | 0.1693 | 0.6331 | 0.4157 |

SPKD halved; structure improved after removing the offset; cross-molecule cosine
moved *toward* the teacher. Respelling invariance (40 peptides × 8 verified
identical SMILES spellings) also moved toward the teacher: margin 0.031 → 0.036
against the teacher's 0.040, with treatment − warm-start self-similarity +0.0179
(p = 1.3e-12).

### Downstream: PAMPA permeability

Full finetuning (their protocol for this benchmark — the entire backbone trains),
clusters 1 and 6 held out, 5 inner folds ensembled, seed 101. Two runs at
different training budgets.

**Run 1** — `max_steps=10000`. Every job stopped on the step cap at ~36 epochs
(28–43 depending on train-set size); `max_epochs=250` was never approached and
`patience=20` never fired.

**Run 2** — `max_epochs=100, max_steps=36000`, ~2.8× the budget. Both numbers were
raised together because in their script `max_steps` sets the LR decay horizon as
well as the hard cap (`total_steps = max(1, int(args.max_steps))`, and argparse
never leaves it `None`, so the steps-per-epoch branch is dead code). Raising only
the cap would leave the LR near peak when early stopping fires.

R² on the held-out cluster, from the 5-fold ensemble:

| model | cluster | run 1 (~36 ep) | run 2 (100 ep) |
|---|---|---|---|
| warm-start | 1 | 0.060 | **0.066** |
| warm-start | 6 | 0.296 | **0.320** |
| treatment | 1 | 0.045 | 0.011 |
| treatment | 6 | 0.232 | 0.269 |

Distillation effect (treatment − warm-start):

| | cluster 1 | cluster 6 |
|---|---|---|
| run 1 | −0.015 | −0.064 |
| run 2 | **−0.055** | **−0.050** |

**Distillation did not help, and more training did not change that.** Both models
improved with the larger budget — warm-start 0.296 → 0.320 on cluster 6, treatment
0.232 → 0.269 — so the extra epochs were doing something, but the gap did not
close. Treatment is behind warm-start on both clusters in both runs, four out of
four. Spearman agrees (run 2: warm-start 0.303 / 0.618 vs treatment 0.247 /
0.598), as does RMSE (0.865 vs 0.897 on cluster 6), so the result is not an
artifact of R²'s variance denominator.

The undertraining caveat raised after run 1 is therefore resolved, and it was not
the explanation.

For reference, the authors' published numbers on the same clusters, recomputed
from their shipped predictions:

| model | cluster 1 | cluster 6 |
|---|---|---|
| their MLM-small (3 runs) | −0.454 ± 0.155 | 0.500 ± 0.012 |
| their MLM-large (3 runs) | −0.309 ± 0.213 | 0.847 ± 0.021 |

Three caveats that still stand:

- **Cluster 1 is noise-dominated.** Their own three runs of an identical model span
  0.38 R² (−0.487, −0.625, −0.250). No model in the comparison predicts it. On
  cluster 1 the individual fold models average R² ≈ −0.14 — worse than predicting
  the mean — and only the ensemble reaches a mildly positive number.
- **Our absolute numbers are below theirs on cluster 6** (0.320 vs 0.500) using the
  *same released weights*, so something in the finetuning protocol differs. Their
  headline figure is also a *pooled* R² over all 6,701 molecules, not per cluster.
- **Single seed.** Given cluster 1's spread, ≥3 seeds are needed before per-cluster
  differences of ~0.05 mean much.

One thing I could not establish: run 2's logs contain no `Trainer.fit stopped`
lines, so whether jobs ended on patience or on the 100-epoch cap is unknown.

### Next: LoRA

`distill/pampa_lora.py` and `kaggelscripts/kaggle_pampa_lora_warmstart.ipynb`
implement the same protocol with the backbone **frozen** and LoRA adapters
(r=16, α=32 on `qkv_proj`, ~2.2% of weights trainable), config lifted verbatim
from their classification script. Head, loss, optimizer, schedule, batch size,
early stopping and fold logic are unchanged; only the trainable parameter set
differs.

This is the more informative test. Full finetuning lets 31.9M parameters reshape a
mediocre representation into a good one, which can mask differences between
backbones — plausibly why the two arms landed so close above. With the trunk
frozen, downstream performance is governed by the quality of the frozen features,
which is exactly what distillation was supposed to improve.

## 8. Corrections

Recorded because they changed conclusions.

**Embedding collapse — retracted.** An earlier round of geometry probes reported
that treatment's embedding space was collapsing (cross-cosine 0.691 vs teacher
0.514) and that SPKD had failed. Both were artifacts. Loading the 337M teacher and
a 32M student **in the same process** changes the student's output despite
byte-identical weights (verified: same `state_dict` hash, cross-cosine 0.5013 vs
0.5566). Their `RotaryPositionalEmbeddings` registers `theta`/`cache` as
non-persistent buffers rebuilt lazily inside `forward()`, and two models of this
family corrupt each other's rotary state — their own code carries a comment about
"corrupted non-persistent buffers". All geometry is now measured one model per
process (`distill/probe_embed.py`).

**Intermediate wrong explanations**, also retracted: padding sensitivity and
batch-composition noise were both proposed and both disproven by controlled tests
(bs=1 through bs=64 give bit-identical results).

**Small-sample instability.** An early respelling result measured on 8 molecules
(28 pairs) gave Spearman swings of ±0.1 that vanished at 1,024 molecules. Any
Gram-based statistic on fewer than ~100 molecules is unreliable.

**fp16 is safe for the teacher.** A measurement suggesting KL 0.236 / 90% top-1
agreement between fp16 and fp32 could not be reproduced under controlled
conditions. With the model dtype verified before and after: max logit difference
0.034, KL 1.1e-06, **100%** top-1 agreement.

---

## 9. Reproducibility notes

Issues in the released repository that must be worked around:

- `training/adapters/` and `training/experiment/` are imported by the regression
  script but ship under `training/02_classification_benchmarks_training_code/`.
  Location matters for correctness, not just importability: `manifest.py` computes
  `REPO_ROOT` as `parents[2]`, which only resolves correctly from
  `training/experiment/`.
- `run_classification_finetuning.sh` calls a script that does not exist and lists
  HuggingFace ids that no longer resolve.
- `--gpu_index` defaults to `None` but is used as `int(args.gpu_index)`.
- Their scripts import `transformers.tokenization_utils_tokenizers.TokenizersBackend`
  at module level — **transformers 5.x only**.
- `peft ≥ 0.17` raises rather than returning False when `torchao < 0.16` is
  installed; uninstalling torchao is the minimal fix.
- `max_steps` below one epoch stops training mid-epoch, so `val_rmse` is never
  logged and EarlyStopping raises.
- Their `ModelCheckpoint` saves the entire 337M module per fold — a 3-seed CellPPD
  run leaves ~21 GB.

---

## 10. Where it stands

Distillation transfers the teacher's behaviour. That is established three
independent ways — cached targets, live teacher on training molecules, and live
teacher on four held-out benchmark sets — and the control arm rules out "more
training" as the explanation.

It has not been shown to make the student **better at anything**. PAMPA has now
been run at two training budgets, and the distilled model is behind its own
starting point on both held-out clusters in both runs. The undertraining
explanation is ruled out.

So the current finding is a negative one, and a fairly clean one: matching a larger
model's token distributions and representation geometry is **not sufficient** to
transfer its downstream capability, at least on this task at this scale. Every
intrinsic measure says the student moved toward the teacher; the benchmark says it
gained nothing by doing so.

What would sharpen it, in order:

1. **LoRA evaluation** (code ready). With the backbone frozen, representation
   quality is what is actually being measured, rather than the finetuner's ability
   to repair it.
2. **≥3 seeds.** Cluster 1's run-to-run spread in the authors' own data is ±0.38 R²,
   so single-seed differences there carry little weight.
3. **THPep and AmpHGT**, where the teacher's margin over fingerprint baselines is
   largest and the distillation signal was strongest (KL 4.3× and 2.1×
   respectively, versus 1.8× on PAMPA).
4. **Reconcile cluster 6 against their published 0.500** using identical weights,
   which would tell us whether our absolute numbers can be quoted at all.
