"""Hyperparameters, with the reasoning attached.

Every number here is either measured on this pipeline or inherited from the
paper's pretraining, and it says which.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Loss weights
# ---------------------------------------------------------------------------
# These are NOT the paper's 0.6 / 0.4. That split assumed an MLM loss starting
# near 6 nats (training from scratch). Ours starts at 0.078, because the student
# is warm-started from their released 32M model and already reproduces the
# teacher's token distributions (91.8% top-1 agreement, measured). Inheriting
# 0.6/0.4 would let MTR dominate KD roughly 8:1 and make SPKD invisible.
#
# Instead each lambda is set so the term contributes its intended SHARE of the
# objective at initialisation:  lambda_k = share_k / L_k(init)
#
#   term   L(init)   lambda   share
#   KD      0.078      4.0     30%
#   MTR     ~1.0       0.3     30%
#   SPKD    ~4e-4     1000     40%
#
# SPKD carries the largest share deliberately. The measurement that motivates it:
# the warm-started student already matches the teacher as a token predictor, so
# KD has little headroom -- but it does NOT match the teacher's representation
# geometry, which is what the paper's downstream gap (R^2 0.13 vs 0.58) is about.
# lambda_sp = 1000 looks extreme but is in line with the SPKD paper's own gamma
# of 3000: the row-normalized Gram difference is intrinsically small.
LAMBDAS = {"kd": 4.0, "mtr": 0.3, "sp": 1000.0}

TARGET_SHARES = {"kd": 0.30, "mtr": 0.30, "sp": 0.40}

# Descriptors are z-scored but not winsorized (observed range -13.25 to +583).
# 0.007% of cells exceed |10|; squared error would let those dominate.
MTR_CLIP = 10.0

# KD temperature. Kept at 1.0 on purpose: at T=1 the cached top-16 probabilities
# are EXACT. Any T != 1 needs renormalising over all 405 logits, and we only
# stored 16 -- so it would redistribute mass into precisely the tail we discarded.
KD_TEMPERATURE = 1.0


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------
# Backbone runs at a third of their 3e-4 pretraining peak. We start from trained
# weights and only spend 640M tokens; a hotter LR erases the initialisation that
# is the entire reason for warm-starting. The MTR head is randomly initialised
# and runs 3x hotter to catch up.
LR_BACKBONE = 1e-4
LR_HEAD = 3e-4

# Inherited from their pretraining (paper section 4.3.4).
BETAS = (0.9, 0.98)
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

WARMUP_STEPS = 2000          # ~5% of the run
MIN_LR_FRAC = 0.1            # cosine floor, as in their schedule

# ~104 molecules/batch at a mean of 158 tokens -> ~19.4k steps/epoch on 2M
# molecules, ~38.8k steps total. Measured peak memory at 2048 tokens was 1.35 GB,
# so 16384 leaves plenty of headroom on a 16 GB T4.
MAX_TOKENS = 16384
EPOCHS = 2

# Epoch 0 uses mask A, epoch 1 uses mask B. Deterministic alternation rather than
# random choice: with 2 epochs, sampling at random leaves half the molecules
# seeing one mask twice and the other never.
MASK_BY_EPOCH = ["a", "b"]


def lr_at(step, total_steps, base_lr):
    """Linear warmup then cosine decay to MIN_LR_FRAC of peak."""
    if step < WARMUP_STEPS:
        return base_lr * (step + 1) / WARMUP_STEPS
    p = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    p = min(1.0, max(0.0, p))
    cos = 0.5 * (1.0 + np.cos(np.pi * p))
    return base_lr * (MIN_LR_FRAC + (1.0 - MIN_LR_FRAC) * cos)


def calibrate(observed, target_shares=None):
    """Recompute lambdas from measured loss magnitudes.

    Call this AFTER WARMUP (step ~2500), not during it. Measured on this
    pipeline: at step 20 L_mtr was 4.3 and calibration set lambda_mtr = 0.07; by
    step 1100 L_mtr had fallen to 0.3. Calibrating early therefore under-weights
    MTR by ~14x for the entire run. Nothing meaningfully learns before warmup
    ends, so any earlier measurement is of an untrained head, not of the term.

    Args:
        observed: {"kd": float, "mtr": float, "sp": float} recent means
    """
    shares = target_shares or TARGET_SHARES
    out = {}
    for k, share in shares.items():
        mag = float(observed.get(k, 0.0))
        out[k] = share / mag if mag > 1e-12 else 0.0
    return out


def describe(observed, lambdas):
    """Human-readable contribution table, for the training log."""
    contrib = {k: lambdas.get(k, 0.0) * float(observed.get(k, 0.0)) for k in observed}
    tot = sum(contrib.values()) or 1.0
    lines = ["%-6s %10s %10s %10s %8s" % ("term", "raw", "lambda", "weighted", "share")]
    for k in observed:
        lines.append("%-6s %10.5f %10.2f %10.5f %7.1f%%"
                     % (k, observed[k], lambdas.get(k, 0.0), contrib[k],
                        100 * contrib[k] / tot))
    return "\n".join(lines)
