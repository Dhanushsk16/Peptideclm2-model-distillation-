"""Pure Hinton knowledge distillation for masked language modelling.

    L_total = (1 - alpha) * L_hard(y_true, sigma(z_s))
            + alpha * T^2 * L_soft(sigma(z_t / T), sigma(z_s / T))

Deliberately simpler than the earlier three-term objective: no descriptor
regression, no similarity-preserving term. Just the true token and the teacher's
softened distribution over it.

Two things this buys over the cached-teacher setup:

  * FULL 405-way distributions. The cache stored only the top 16 logits, which
    captured 96.2% of the mass on average but 80% at the 5th percentile -- i.e.
    it truncated hardest exactly where the teacher was most uncertain and the
    soft target carried the most information. Nothing is truncated here.

  * A FRESH MASK every epoch. Cached targets are only valid for the mask they
    were computed under, so the earlier run reused one fixed mask per molecule.
    A live teacher re-masks for free.

The cost is compute: the teacher forward runs on every batch, and at 337M vs 32M
it dominates the step (~78% of the FLOPs).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def hinton_kd_loss(student_logits, teacher_logits, labels, temperature=4.0,
                   alpha=0.95):
    """Return (total, hard, soft) at the masked positions.

    Args:
        student_logits: (N, V) student logits, masked positions only
        teacher_logits: (N, V) teacher logits, same positions
        labels:         (N,)   true token ids at those positions
        temperature:    T in the formula. Applied to BOTH logit sets.
        alpha:          weight on the soft term.

    The T^2 factor is Hinton's gradient-scale correction, not cosmetic. Softening
    by T shrinks the soft-target gradients by roughly 1/T^2, so without it the
    soft term's contribution would collapse as T rises and alpha would no longer
    mean what it says. With T=4 and alpha=0.95 the effective weights are
    0.05 on hard and 0.95 * 16 = 15.2 on soft -- the soft term dominates by
    design, which is the point of pure KD.
    """
    s = student_logits.float()
    t = teacher_logits.float()

    # Hard term: ordinary cross-entropy against the token that was actually there.
    hard = F.cross_entropy(s, labels)

    # Soft term: KL(teacher || student) at temperature T over the FULL vocabulary.
    # F.kl_div(input=log q, target=p) computes sum p * (log p - log q), so passing
    # the student as `input` and the teacher as `target` gives KL(p_t || p_s) --
    # the mode-covering direction, which is what KD wants.
    s_log = F.log_softmax(s / temperature, dim=-1)
    t_prob = F.softmax(t / temperature, dim=-1)
    soft = F.kl_div(s_log, t_prob, reduction="batchmean")

    total = (1.0 - alpha) * hard + alpha * (temperature ** 2) * soft
    return total, hard.detach(), soft.detach()


@torch.no_grad()
def agreement(student_logits, teacher_logits, labels):
    """Diagnostics worth logging every step: how often the student picks the
    teacher's top token, and how often each picks the true one."""
    s_top = student_logits.argmax(-1)
    t_top = teacher_logits.argmax(-1)
    return {
        "agree_teacher": (s_top == t_top).float().mean().item(),
        "acc_student": (s_top == labels).float().mean().item(),
        "acc_teacher": (t_top == labels).float().mean().item(),
    }
