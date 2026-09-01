"""The three distillation loss terms.

Every term is written against the cache format produced by the teacher-caching
notebook, documented in that cache's manifest.json:

    topv  fp16   log-probabilities, so p = exp(topv)   (NOT raw logits)
    topi  int16  vocab indices for those probabilities
    lse   fp32   full-vocab logsumexp; raw logit = topv + lse
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Term 1: soft MLM knowledge distillation
# --------------------------------------------------------------------------
def kd_loss(student_logits, topi, topv, reduction="mean"):
    """KL(teacher || student) over 17 buckets: the teacher's top-16 plus a lump
    for everything else.

    We only cached the top 16 of 405 logits (96.2% of the mass on average, 80%
    at the 5th percentile). The 17th bucket keeps the discarded tail honest --
    without it the student is fitted to a distribution that sums to 0.96 and
    pretends the rest does not exist.

    Args:
        student_logits: (N, V) student logits at the masked positions
        topi:           (N, K) teacher's top-K vocab indices
        topv:           (N, K) teacher's log-probabilities for those indices

    Returns:
        scalar loss (or (N,) if reduction="none")
    """
    p = topv.float().exp()                                  # (N, K) teacher probs
    p_other = (1.0 - p.sum(-1)).clamp_min(0.0)              # clamp: fp16 rounding
                                                            # can make this ~-3e-5
    logq_all = F.log_softmax(student_logits.float(), dim=-1)
    logq = logq_all.gather(-1, topi.long())                 # (N, K)
    q_other = (1.0 - logq.exp().sum(-1)).clamp_min(1e-9)

    eps = 1e-9
    kl = (p * (p.clamp_min(eps).log() - logq)).sum(-1)
    kl = kl + p_other * (p_other.clamp_min(eps).log() - q_other.log())

    if reduction == "none":
        return kl
    return kl.mean() if reduction == "mean" else kl.sum()


# --------------------------------------------------------------------------
# Term 2: multi-task regression onto RDKit descriptors
# --------------------------------------------------------------------------
def mtr_loss(pred, target, clip=10.0):
    """MSE against the 99 pre-normalized RDKit descriptors.

    Not distillation: these are ground truth from the subset parquet, and the
    teacher is not involved. Included because the paper's own scaling result says
    a 32M model needs explicit physicochemical supervision (R^2 0.38 vs 0.13 on
    permeability) while a 337M one does not.

    The clip is load-bearing. The stored descriptors are z-scored but NOT
    winsorized: the observed range is -13.25 to +583. Squared error would let a
    single cell at 583 contribute ~340,000x a typical one. Only 0.007% of cells
    exceed |10|, so clipping there costs nothing and removes the pathology.
    """
    return F.mse_loss(pred.float(), target.float().clamp(-clip, clip))


# --------------------------------------------------------------------------
# Term 3: similarity-preserving KD
# --------------------------------------------------------------------------
def spkd_loss(student_pool, teacher_pool):
    """Match the within-batch Gram matrices of student and teacher embeddings.

    Direct feature matching is impossible here: independently trained models sit
    in unrelated bases. We measured cosine(MLM-large, MTR-large) = 0.004 on the
    same molecule -- essentially orthogonal. Gram matrices sidestep that, because
    Z @ Z.T is invariant to any orthogonal rotation of the feature space:

        (ZR)(ZR).T = Z R R.T Z.T = Z Z.T

    It also erases the width mismatch: student (B,512) and teacher (B,1024) both
    give (B,B), so no projection head is needed.

    The claim being transferred is relational -- "if the teacher puts molecules i
    and j close together, so must the student" -- which is exactly what the
    paper's t-SNE and class-separation figures are about.
    """
    def gram(z):
        g = z.float() @ z.float().t()          # (B, B)
        return F.normalize(g, p=2, dim=1)      # row-normalize, as in Tung & Mori

    gs, gt = gram(student_pool), gram(teacher_pool)
    return (gs - gt).pow(2).mean()


# --------------------------------------------------------------------------
def total_loss(parts, lambdas):
    """Weighted sum. Kept separate from the terms so the training loop can log
    each one unweighted -- with a single scalar you cannot tell whether the
    student learned from the teacher's distributions, from RDKit, or from the
    relational structure, which is the whole question the run exists to answer.
    """
    return sum(lambdas[k] * parts[k] for k in parts if k in lambdas)
