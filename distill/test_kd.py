"""Correctness tests for the live-teacher KD pipeline. Run before training.

Checks properties, not just "does it run". The one that matters most is the
rotary-buffer check: live teaching puts teacher and student in one process, which
is the configuration that silently corrupted them before.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from data_live import LiveStream, span_mask_positions, build_batches
from losses_kd import agreement, hinton_kd_loss
from student import patch_sdpa_dtype
from train_kd import load_pair, logits_of, refresh_rope, verify_rope

OK, FAIL = "  ok  ", " FAIL "

results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print("[%s] %-50s %s" % (OK if cond else FAIL, name, detail))


# ------------------------------------------------------------ solo reference
# The rotary check needs the student's output measured WITHOUT the teacher in the
# process, which cannot be done once the teacher is loaded. So the test re-invokes
# itself with --solo-ref first and captures the number.
#
# This used to be a hardcoded constant measured on one machine. That was wrong:
# the value depends on the GPU's kernels, so the same correct code produces a
# slightly different number on a T4 than on an RTX 3050, and the check would have
# failed on Kaggle for reasons having nothing to do with corruption. Measuring it
# here makes the comparison exact and machine-independent.


def solo_cosine(student_path, device, smis):
    """Mean pairwise cosine of the student's mean-pooled embeddings."""
    patch_sdpa_dtype()
    tok = AutoTokenizer.from_pretrained(student_path, trust_remote_code=True)
    m = AutoModel.from_pretrained(student_path, trust_remote_code=True).to(device).eval()
    refresh_rope(m, device)
    verify_rope(m, "student(solo)")
    b = tok(smis, return_tensors="pt", padding=True, truncation=True, max_length=512)
    b = {k: v.to(device) for k, v in b.items()}
    with torch.no_grad():
        o = m(**b)
    p = o["mean_pool"] if isinstance(o, dict) else o.mean_pool
    n = F.normalize(p.float(), dim=1)
    return (n @ n.t())[np.triu_indices(len(smis), 1)].mean().item()


def solo_reference(argv_student, subset, device, limit, max_tokens):
    """Run solo_cosine in a FRESH process and return the value it printed."""
    cmd = [sys.executable, os.path.abspath(__file__), "--solo-ref",
           "--subset", subset, "--teacher", "-", "--student", argv_student,
           "--device", device, "--limit", str(limit), "--max-tokens", str(max_tokens)]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.abspath(__file__)) or ".")
    for line in r.stdout.splitlines():
        if line.startswith("SOLO_COSINE "):
            return float(line.split()[1])
    print(r.stdout[-1500:]); print(r.stderr[-1500:])
    raise RuntimeError("solo reference subprocess failed (exit %d)" % r.returncode)


# ---------------------------------------------------------------- loss
def test_loss():
    print("\n=== KD loss ===")
    g = torch.Generator().manual_seed(0)
    N, V, T, A = 512, 405, 4.0, 0.95
    tl = torch.randn(N, V, generator=g) * 3
    labels = tl.argmax(-1)

    tot, hard, soft = hinton_kd_loss(tl.clone(), tl, labels, T, A)
    check("soft ~ 0 when student == teacher", soft.item() < 1e-6, "%.2e" % soft.item())
    check("hard > 0 even then (labels != argmax always)", hard.item() >= 0,
          "%.4f" % hard.item())

    _, _, soft_rand = hinton_kd_loss(torch.randn(N, V, generator=g), tl, labels, T, A)
    check("soft > 0 for a wrong student", soft_rand.item() > 0, "%.4f" % soft_rand.item())
    check("soft(random) > soft(exact)", soft_rand.item() > soft.item())

    # The T^2 factor must actually be applied.
    t2, _, s2 = hinton_kd_loss(torch.randn(N, V, generator=g), tl, labels, T, 1.0)
    check("total == alpha*T^2*soft when alpha=1", abs(t2.item() - (T ** 2) * s2.item()) < 1e-4,
          "%.4f vs %.4f" % (t2.item(), (T ** 2) * s2.item()))

    t3, h3, _ = hinton_kd_loss(torch.randn(N, V, generator=g), tl, labels, T, 0.0)
    check("total == hard when alpha=0", abs(t3.item() - h3.item()) < 1e-5)

    # KL direction: KL(teacher || student), not the reverse.
    p = torch.tensor([[0.7, 0.2, 0.1]]).log()
    q = torch.tensor([[0.1, 0.2, 0.7]]).log()
    manual = (p.exp() * (p - q)).sum().item()
    got = F.kl_div(q, p.exp(), reduction="batchmean").item()
    check("KL direction is KL(teacher || student)", abs(manual - got) < 1e-6,
          "%.5f" % got)

    # Temperature must soften: higher T -> lower KL between the same two models.
    s = torch.randn(N, V, generator=g)
    kls = [hinton_kd_loss(s, tl, labels, t, 1.0)[2].item() for t in (1.0, 2.0, 4.0, 8.0)]
    check("higher T softens the target", all(a > b for a, b in zip(kls, kls[1:])),
          " > ".join("%.4f" % k for k in kls))

    eff_hard, eff_soft = 1 - A, A * T ** 2
    check("effective weights match the formula",
          abs(eff_hard - 0.05) < 1e-9 and abs(eff_soft - 15.2) < 1e-9,
          "hard %.2f soft %.2f" % (eff_hard, eff_soft))


# ---------------------------------------------------------------- masking
def test_masking():
    print("\n=== masking ===")
    rng = np.random.default_rng(0)
    for L in (20, 60, 158, 340, 512):
        got = [len(span_mask_positions(L, rng)) for _ in range(200)]
        tgt = int(L * 0.25)
        check("len %4d -> ~25%% masked" % L, abs(np.mean(got) - tgt) <= 1,
              "target %d, got %.1f" % (tgt, np.mean(got)))
    p = span_mask_positions(200, rng)
    check("no duplicate positions", len(p) == len(set(p)))
    check("positions in range", min(p) >= 0 and max(p) < 200)

    # Different epochs must produce different masks -- the whole point of a live
    # teacher is that the mask is no longer frozen.
    a = set(span_mask_positions(200, np.random.default_rng(1)))
    b = set(span_mask_positions(200, np.random.default_rng(2)))
    check("mask varies with rng", a != b, "%d shared of %d" % (len(a & b), len(a)))

    lengths = np.concatenate([np.full(200, 23), np.full(200, 340)])
    batches = build_batches(lengths, 4096)
    check("batches respect the token budget",
          all(lengths[b].max() * len(b) <= 4096 for b in batches))
    check("every molecule appears once",
          sorted(i for b in batches for i in b) == list(range(len(lengths))))
    pad = [1 - lengths[b].sum() / (lengths[b].max() * len(b)) for b in batches]
    check("length bucketing keeps padding low", np.mean(pad) < 0.05,
          "%.2f%% padding" % (100 * np.mean(pad)))


# ---------------------------------------------------------------- data
def test_data(stream):
    print("\n=== data ===")
    b = next(iter(stream.iter_epoch(0)))
    T = b["input_ids"].shape[1]
    check("masked positions inside sequence", int(b["cols"].max()) < T)
    check("[CLS] never masked", int(b["cols"].min()) >= 1)
    check("no masked position on padding",
          bool((b["attention_mask"][b["rows"], b["cols"]] == 1).all()))
    check("MASK written at every position",
          bool((b["input_ids"][b["rows"], b["cols"]] == stream.MASK).all()))
    check("labels captured before masking", bool((b["labels"] != stream.MASK).all()))
    real = int(b["attention_mask"].sum()) - 2 * b["input_ids"].shape[0]
    check("masking rate ~ 0.25", 0.20 < b["cols"].shape[0] / real < 0.28,
          "%.4f" % (b["cols"].shape[0] / real))

    # Epoch 0 and epoch 1 must mask differently.
    b0 = next(iter(stream.iter_epoch(0)))
    b1 = next(iter(stream.iter_epoch(1)))
    same = (b0["input_ids"].shape == b1["input_ids"].shape and
            torch.equal(b0["input_ids"], b1["input_ids"]))
    check("epoch 1 uses a different mask than epoch 0", not same)


# ---------------------------------------------------------------- rotary
def test_rotary(teacher, student, tok, device, smis, solo_ref):
    """The check this pipeline exists to pass.

    A student loaded ALONE is the reference. If loading it alongside the teacher
    changes its outputs, the earlier corruption is back.
    """
    print("\n=== rotary integrity (teacher + student in one process) ===")
    check("teacher rotary verified", verify_rope(teacher, "teacher") > 0)
    check("student rotary verified", verify_rope(student, "student") > 0)

    b = tok(smis, return_tensors="pt", padding=True, truncation=True, max_length=512)
    b = {k: v.to(device) for k, v in b.items()}
    with torch.no_grad():
        pooled = student(**b)
    pooled = pooled["mean_pool"] if isinstance(pooled, dict) else pooled.mean_pool
    n = F.normalize(pooled.float(), dim=1)
    cross = (n @ n.t())[np.triu_indices(len(smis), 1)].mean().item()
    check("student outputs finite", bool(torch.isfinite(pooled).all()))
    print("       with teacher resident %.6f | student alone %.6f (this machine)"
          % (cross, solo_ref))
    # Bit-identical, not merely close. The corruption shifted this by ~0.055
    # (0.5013 -> 0.5566 when it was first found), so a loose tolerance would
    # catch it too -- but there is no legitimate reason for ANY difference, and
    # a tight bound also catches smaller variants of the same failure.
    check("matches student-alone reference (no corruption)",
          abs(cross - solo_ref) < 1e-6, "delta %.3e" % abs(cross - solo_ref))
    return cross


def test_step(teacher, student, stream, device, temperature, alpha):
    print("\n=== forward / backward ===")
    opt = torch.optim.AdamW(student.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(device.split(":")[0], enabled=(device != "cpu"))
    student.train()
    hist, gn = [], None
    for i, b in enumerate(stream.iter_epoch(0)):
        if i >= 3:
            break
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast(device.split(":")[0], dtype=torch.float16,
                            enabled=(device != "cpu")):
            with torch.no_grad():
                tl = logits_of(teacher(input_ids=b["input_ids"],
                                       attention_mask=b["attention_mask"]))[b["rows"], b["cols"]]
            sl = logits_of(student(input_ids=b["input_ids"],
                                   attention_mask=b["attention_mask"]))[b["rows"], b["cols"]]
            loss, hard, soft = hinton_kd_loss(sl, tl, b["labels"], temperature, alpha)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        ag = agreement(sl, tl, b["labels"])
        hist.append(loss.item())
        print("   step %d B=%3d T=%3d masked=%5d | loss %.4f hard %.4f soft %.5f "
              "| agree %.3f | gn %.2f"
              % (i, b["input_ids"].shape[0], b["input_ids"].shape[1],
                 b["rows"].numel(), loss.item(), hard.item(), soft.item(),
                 ag["agree_teacher"], gn))

    check("three steps completed", len(hist) == 3)
    check("losses finite", all(np.isfinite(hist)))
    check("gradient norm finite and > 0", bool(torch.isfinite(gn)) and gn > 0,
          "%.3f" % gn)
    nog = [n for n, p in student.named_parameters() if p.requires_grad and p.grad is None]
    check("every student parameter got a gradient", not nog, "%d missing" % len(nog))
    check("teacher stayed frozen",
          all(not p.requires_grad for p in teacher.parameters()))
    if device != "cpu":
        print("   peak GPU memory: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))


def test_overfit(teacher, student, stream, device, temperature, alpha, steps=60):
    """On one batch the loss must fall a long way. If it does not, inputs and
    targets are misaligned somewhere."""
    print("\n=== overfit one batch ===")
    b = {k: v.to(device) for k, v in next(iter(stream.iter_epoch(0))).items()}
    opt = torch.optim.AdamW(student.parameters(), lr=3e-4)
    student.train()
    first = last = None
    with torch.no_grad():
        tl = logits_of(teacher(input_ids=b["input_ids"],
                               attention_mask=b["attention_mask"]))[b["rows"], b["cols"]].float()
    for s in range(steps):
        sl = logits_of(student(input_ids=b["input_ids"],
                               attention_mask=b["attention_mask"]))[b["rows"], b["cols"]]
        loss, _, soft = hinton_kd_loss(sl, tl, b["labels"], temperature, alpha)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        first = loss.item() if s == 0 else first
        last = loss.item()
        if s % 20 == 0 or s == steps - 1:
            print("   step %3d  loss %.4f  soft %.5f" % (s, loss.item(), soft.item()))
    check("loss drops on a single batch", last < first * 0.7,
          "%.4f -> %.4f (%.0f%% down)" % (first, last, 100 * (1 - last / first)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--solo-ref", action="store_true",
                    help="internal: print the student-alone cosine and exit")
    a = ap.parse_args()

    if a.device != "cpu":
        p = torch.cuda.get_device_properties(0)
        print("device: %s  %.1f GB" % (p.name, p.total_memory / 1e9))

    tok = AutoTokenizer.from_pretrained(a.student, trust_remote_code=True)
    stream = LiveStream(a.subset, tok, max_tokens=a.max_tokens, seed=0,
                        val_molecules=0, split="train", limit=a.limit)
    smis = list(stream.smiles[:16])

    if a.solo_ref:                       # child process: no teacher is ever loaded
        print("SOLO_COSINE %.9f" % solo_cosine(a.student, a.device, smis))
        return

    test_loss()
    test_masking()
    print("\nstream: %d molecules" % len(stream))
    test_data(stream)

    solo_ref = solo_reference(a.student, a.subset, a.device, a.limit, a.max_tokens)
    teacher, student = load_pair(a.teacher, a.student, a.device)
    test_rotary(teacher, student, tok, a.device, smis, solo_ref)
    test_step(teacher, student, stream, a.device, a.temperature, a.alpha)
    test_overfit(teacher, student, stream, a.device, a.temperature, a.alpha)

    print("\n%d/%d checks passed" % (sum(results), len(results)))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
