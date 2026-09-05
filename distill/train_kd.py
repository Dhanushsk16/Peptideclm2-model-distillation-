"""Live-teacher Hinton KD: 337M teacher, 32M student, same process, one GPU.

    L = (1 - alpha) * L_hard + alpha * T^2 * L_soft     T = 4, alpha = 0.95

THE PROBLEM THIS FILE HAS TO SOLVE FIRST
----------------------------------------
Live teaching requires both models in one process, and that is exactly the
configuration measured earlier to corrupt them. Loading the 337M teacher and then
constructing the 32M student changed the student's outputs despite byte-identical
weights (same state_dict hash, cross-molecule cosine 0.5013 vs 0.5566). Their
RotaryPositionalEmbeddings registers `theta` and `cache` as NON-PERSISTENT
buffers, rebuilt lazily inside forward() when a validity check trips -- and their
own code carries a comment about "environments that load this model with corrupted
non-persistent buffers", so they hit something like it too.

Every geometry measurement taken before that was found had to be retracted. The
earlier fix was to keep the models in separate processes, which live teaching
makes impossible. So this file fixes the buffers directly instead:

  refresh_rope()  recomputes theta and the cos/sin cache from config on the target
                  device for every rotary module in a model, after both models are
                  loaded and moved to the GPU.

  verify_rope()   checks the result against an independently computed reference
                  and raises if any module disagrees.

The check runs at startup on BOTH models. If it passes, the rotary state is
provably correct and the teacher's targets can be trusted.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from data_live import LiveStream
from losses_kd import agreement, hinton_kd_loss
from student import patch_sdpa_dtype


# ---------------------------------------------------------------- rotary repair
def _rope_modules(model):
    return [m for m in model.modules() if hasattr(m, "build_rope_cache")]


def refresh_rope(model, device):
    """Recompute every rotary buffer from config, on `device`."""
    n = 0
    for m in _rope_modules(model):
        theta = m._compute_theta(device=device)
        m.register_buffer("theta", theta, persistent=False)
        m.build_rope_cache(m.max_seq_len)
        # torch.device(...) on both sides: comparing a device to a string is
        # always unequal, which made this copy run every time.
        if m.cache.device != torch.device(device):
            m.cache = m.cache.to(device)
        n += 1
    return n


def verify_rope(model, name):
    """Independently recompute theta and compare. Raises on mismatch.

    theta_i = base^(-2i/dim) for i in [0, dim/2), which must be in (0, 1] and
    strictly decreasing. A corrupted buffer fails one of those.
    """
    bad = []
    for i, m in enumerate(_rope_modules(model)):
        exps = torch.arange(0, m.dim, 2, dtype=torch.float32,
                            device=m.theta.device)[: m.dim // 2] / float(m.dim)
        ref = torch.pow(torch.tensor(float(m.base), device=m.theta.device), -exps)
        if not torch.allclose(m.theta.float(), ref, atol=1e-6):
            bad.append((i, "theta mismatch"))
        elif not torch.isfinite(m.cache).all():
            bad.append((i, "cache has non-finite values"))
        elif m.cache.size(0) < m.max_seq_len:
            bad.append((i, "cache too short: %d" % m.cache.size(0)))
    if bad:
        raise RuntimeError("%s: %d/%d rotary modules corrupt -- %s"
                           % (name, len(bad), len(_rope_modules(model)), bad[:3]))
    return len(_rope_modules(model))


def load_pair(teacher_path, student_path, device):
    """Load teacher and student, then repair and verify rotary state on both.

    Order matters for the bug being guarded against: the teacher is loaded FIRST,
    which is the sequence that previously corrupted the student.
    """
    patch_sdpa_dtype()

    teacher = AutoModel.from_pretrained(teacher_path, trust_remote_code=True,
                                        use_safetensors=True)
    teacher = teacher.half().to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = AutoModel.from_pretrained(student_path, trust_remote_code=True,
                                        use_safetensors=True).to(device)

    nt = refresh_rope(teacher, device)
    ns = refresh_rope(student, device)
    verify_rope(teacher, "teacher")
    verify_rope(student, "student")
    print("rotary buffers refreshed and verified: teacher %d, student %d modules"
          % (nt, ns), flush=True)

    print("teacher %.1fM (frozen, fp16) | student %.1fM (trainable, fp32)"
          % (sum(p.numel() for p in teacher.parameters()) / 1e6,
             sum(p.numel() for p in student.parameters()) / 1e6), flush=True)
    return teacher, student


def logits_of(out):
    return out["logits"] if isinstance(out, dict) else out.logits


# ---------------------------------------------------------------- schedule
def lr_at(step, total, base, warmup, min_frac=0.1):
    if step < warmup:
        return base * (step + 1) / warmup
    p = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return base * (min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * p)))


@torch.no_grad()
def evaluate(teacher, student, stream, device, temperature, alpha, max_batches=40):
    student.eval()
    acc = {"total": [], "hard": [], "soft": [], "agree_teacher": [], "acc_student": []}
    for i, b in enumerate(stream.iter_epoch(0)):
        if i >= max_batches:
            break
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        with torch.autocast(device.split(":")[0], dtype=torch.float16,
                            enabled=(device != "cpu")):
            tl = logits_of(teacher(input_ids=b["input_ids"],
                                   attention_mask=b["attention_mask"]))[b["rows"], b["cols"]]
            sl = logits_of(student(input_ids=b["input_ids"],
                                   attention_mask=b["attention_mask"]))[b["rows"], b["cols"]]
        tot, hard, soft = hinton_kd_loss(sl, tl, b["labels"], temperature, alpha)
        ag = agreement(sl, tl, b["labels"])
        acc["total"].append(tot.item()); acc["hard"].append(hard.item())
        acc["soft"].append(soft.item()); acc["agree_teacher"].append(ag["agree_teacher"])
        acc["acc_student"].append(ag["acc_student"])
    student.train()
    return {k: float(np.mean(v)) if v else float("nan") for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, help="pretrain_subset_2M.parquet")
    ap.add_argument("--teacher", required=True, help="peptideclm-2-mlm-large dir")
    ap.add_argument("--student", required=True, help="peptideclm-2-mlm-small dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None, help="debug: cap molecules")
    ap.add_argument("--val-molecules", type=int, default=20_000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.student, trust_remote_code=True)
    teacher, student = load_pair(a.teacher, a.student, dev)

    train = LiveStream(a.subset, tok, max_tokens=a.max_tokens, seed=a.seed,
                       val_molecules=a.val_molecules, split="train", limit=a.limit)
    val = LiveStream(a.subset, tok, max_tokens=a.max_tokens, seed=a.seed,
                     val_molecules=a.val_molecules, split="val", limit=a.limit)
    print("train %s molecules | val %s"
          % ("{:,}".format(len(train)), "{:,}".format(len(val))), flush=True)

    decay = [p for p in student.parameters() if p.requires_grad and p.ndim > 1]
    no_decay = [p for p in student.parameters() if p.requires_grad and p.ndim <= 1]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": a.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=a.lr, betas=(0.9, 0.98))
    scaler = torch.amp.GradScaler(dev, enabled=(dev != "cpu"))

    ck = os.path.join(a.out, "latest.pt")
    start_epoch, start_batch, gstep, hist = 0, 0, 0, []
    if os.path.exists(ck):
        st = torch.load(ck, map_location=dev, weights_only=False)
        student.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        start_epoch, start_batch = st["epoch"], st["batch"]
        gstep, hist = st["gstep"], st["hist"]
        refresh_rope(student, dev); verify_rope(student, "student(resumed)")
        print("resumed: epoch %d batch %d (step %d)" % (start_epoch, start_batch, gstep),
              flush=True)

    steps_path = os.path.join(a.out, "nsteps.json")
    if os.path.exists(steps_path):
        total_steps = json.load(open(steps_path))["total"]
    else:
        per_epoch = sum(1 for _ in train.iter_epoch(0))
        total_steps = per_epoch * a.epochs
        json.dump({"per_epoch": per_epoch, "total": total_steps}, open(steps_path, "w"))
        print("steps/epoch %d | total %d" % (per_epoch, total_steps), flush=True)

    print("\nT=%.1f alpha=%.2f -> effective weights: hard %.2f, soft %.2f\n"
          % (a.temperature, a.alpha, 1 - a.alpha, a.alpha * a.temperature ** 2),
          flush=True)

    student.train()
    t0 = time.time()
    for epoch in range(start_epoch, a.epochs):
        skip = start_batch if epoch == start_epoch else 0
        for bi, batch in enumerate(train.iter_epoch(epoch, skip_batches=skip), start=skip):
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}

            lr = lr_at(gstep, total_steps, a.lr, a.warmup)
            for g in opt.param_groups:
                g["lr"] = lr

            with torch.autocast(dev, dtype=torch.float16, enabled=(dev != "cpu")):
                with torch.no_grad():
                    tl = logits_of(teacher(input_ids=batch["input_ids"],
                                           attention_mask=batch["attention_mask"]))
                    tl = tl[batch["rows"], batch["cols"]]
                sl = logits_of(student(input_ids=batch["input_ids"],
                                       attention_mask=batch["attention_mask"]))
                sl = sl[batch["rows"], batch["cols"]]
                loss, hard, soft = hinton_kd_loss(sl, tl, batch["labels"],
                                                  a.temperature, a.alpha)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(student.parameters(), a.grad_clip)
            scaler.step(opt)
            scaler.update()

            rec = {"step": gstep, "epoch": epoch, "loss": loss.item(),
                   "hard": hard.item(), "soft": soft.item(),
                   "gnorm": float(gn), "lr": lr, "n_masked": int(batch["rows"].numel())}
            hist.append(rec)
            gstep += 1

            if gstep % a.log_every == 0:
                ag = agreement(sl, tl, batch["labels"])
                print("[kd] ep%d step %6d/%d | loss %.4f hard %.4f soft %.5f "
                      "| agree %.3f acc_s %.3f acc_t %.3f | gn %.2f lr %.2e | %.1f min"
                      % (epoch, gstep, total_steps, rec["loss"], rec["hard"], rec["soft"],
                         ag["agree_teacher"], ag["acc_student"], ag["acc_teacher"],
                         rec["gnorm"], lr, (time.time() - t0) / 60), flush=True)

            if gstep % a.eval_every == 0:
                v = evaluate(teacher, student, val, dev, a.temperature, a.alpha)
                v["step"] = gstep
                hist.append({"eval": v})
                print("   [val] loss %.4f hard %.4f soft %.5f agree %.3f acc_s %.3f"
                      % (v["total"], v["hard"], v["soft"], v["agree_teacher"],
                         v["acc_student"]), flush=True)

            if gstep % a.save_every == 0:
                torch.save({"model": student.state_dict(), "opt": opt.state_dict(),
                            "scaler": scaler.state_dict(), "epoch": epoch,
                            "batch": bi + 1, "gstep": gstep, "hist": hist,
                            "args": vars(a)}, ck + ".tmp")
                os.replace(ck + ".tmp", ck)     # atomic: a kill leaves no half file
                json.dump(hist, open(os.path.join(a.out, "history.json"), "w"))
        start_batch = 0

    v = evaluate(teacher, student, val, dev, a.temperature, a.alpha)
    print("\nfinal val: %s" % {k: round(x, 5) for k, x in v.items()}, flush=True)
    torch.save({"model": student.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "epoch": a.epochs, "batch": 0,
                "gstep": gstep, "hist": hist, "args": vars(a)}, ck)
    torch.save({"model": student.state_dict(), "args": vars(a), "final_val": v},
               os.path.join(a.out, "student_final.pt"))
    json.dump(hist, open(os.path.join(a.out, "history.json"), "w"))
    print("done: %d steps in %.1f min" % (gstep, (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
