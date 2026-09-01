"""Distillation training. One process per GPU, one arm per process.

    treatment   KD + MTR + SPKD          (teacher signal)
    control     hard-label MLM + MTR     (no teacher)

Both arms share the same init, data, schedule and step count. The ONLY difference
is whether the teacher's distributions and geometry are in the loss, which is what
makes the delta attributable to distillation. Without the control, "our student
beats their released 32M" is unfalsifiable -- the gain could just be 640M extra
tokens of training.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import config
from data import ShardStream, SubsetTable
from losses import kd_loss, mtr_loss, spkd_loss
from student import Student


def build_parts(arm, out, batch):
    """Loss terms for one arm. Same MTR term in both, by design."""
    sel = out["logits"][batch["rows"], batch["cols"]]
    parts = {"mtr": mtr_loss(out["mtr"], batch["desc"], clip=config.MTR_CLIP)}
    if arm == "treatment":
        parts["kd"] = kd_loss(sel, batch["topi"], batch["topv"])
        parts["sp"] = spkd_loss(out["mean_pool"], batch["teacher_pool"])
    else:
        # Hard labels at the same masked positions: identical task, one-hot
        # supervision instead of the teacher's distribution.
        parts["mlm"] = F.cross_entropy(sel.float(), batch["labels"])
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["treatment", "control"], required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--max-tokens", type=int, default=config.MAX_TOKENS)
    ap.add_argument("--shard-limit", type=int, default=None)
    # After warmup, not during it. Warmup is 2000 steps, so nothing has
    # actually learned before then: calibrating at step 20 saw L_mtr = 4.3
    # (untrained head) and set lambda_mtr = 0.07, but by step 1100 L_mtr had
    # fallen to 0.3 -- a 14x error that would have under-weighted MTR for the
    # whole run. Measured on this pipeline.
    ap.add_argument("--calibrate-at", type=int, default=2500)
    ap.add_argument("--calibrate-window", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.init, trust_remote_code=True)
    table = SubsetTable(a.subset)
    model = Student(a.init).to(dev)
    opt = torch.optim.AdamW(
        model.param_groups(config.LR_BACKBONE, config.LR_HEAD, config.WEIGHT_DECAY),
        betas=config.BETAS)
    scaler = torch.amp.GradScaler(dev, enabled=(dev != "cpu"))

    # Lambdas. The control arm has no teacher terms, so its MLM weight takes the
    # share KD+SPKD hold in the treatment arm (0.30 + 0.40 = 0.70), keeping MTR
    # identical across arms. Both are re-derived from measured magnitudes at
    # --calibrate-at, because the randomly-initialised MTR head starts near 4.7
    # rather than its converged ~1.0.
    if a.arm == "treatment":
        lam = dict(config.LAMBDAS)
        shares = dict(config.TARGET_SHARES)
    else:
        lam = {"mlm": 1.0, "mtr": config.LAMBDAS["mtr"]}
        shares = {"mlm": 0.70, "mtr": 0.30}

    ck = os.path.join(a.out, "latest.pt")
    start_epoch, start_batch, gstep, hist = 0, 0, 0, []
    if os.path.exists(ck):
        st = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        start_epoch, start_batch = st["epoch"], st["batch"]
        gstep, hist, lam = st["gstep"], st["hist"], st["lam"]
        print("resumed: epoch %d batch %d (step %d)" % (start_epoch, start_batch, gstep),
              flush=True)

    # Step count is needed up front for the cosine schedule. Derived from the
    # first epoch's actual batching rather than guessed.
    probe = ShardStream(a.cache, table, tok, mask=config.MASK_BY_EPOCH[0],
                        max_tokens=a.max_tokens, seed=a.seed, shard_limit=a.shard_limit)
    if os.path.exists(os.path.join(a.out, "nsteps.json")):
        total_steps = json.load(open(os.path.join(a.out, "nsteps.json")))["total"]
    else:
        per_epoch = sum(1 for _ in probe.iter_epoch(0))
        total_steps = per_epoch * a.epochs
        json.dump({"per_epoch": per_epoch, "total": total_steps},
                  open(os.path.join(a.out, "nsteps.json"), "w"))
        print("steps/epoch %d | total %d" % (per_epoch, total_steps), flush=True)

    t0 = time.time()
    for epoch in range(start_epoch, a.epochs):
        mask = config.MASK_BY_EPOCH[epoch % len(config.MASK_BY_EPOCH)]
        stream = ShardStream(a.cache, table, tok, mask=mask, max_tokens=a.max_tokens,
                             seed=a.seed, shard_limit=a.shard_limit)
        print("epoch %d using mask %s" % (epoch, mask.upper()), flush=True)

        skip = start_batch if epoch == start_epoch else 0
        for bi, batch in enumerate(stream.iter_epoch(epoch, skip_batches=skip), start=skip):
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}

            lr_b = config.lr_at(gstep, total_steps, config.LR_BACKBONE)
            lr_h = config.lr_at(gstep, total_steps, config.LR_HEAD)
            for g in opt.param_groups[:2]:
                g["lr"] = lr_b
            opt.param_groups[2]["lr"] = lr_h

            with torch.autocast(dev, dtype=torch.float16, enabled=(dev != "cpu")):
                out = model(batch["input_ids"], batch["attention_mask"])
                parts = build_parts(a.arm, out, batch)
                loss = sum(lam[k] * parts[k] for k in parts)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            scaler.step(opt)
            scaler.update()

            raw = {k: v.detach().item() for k, v in parts.items()}
            hist.append({"step": gstep, "epoch": epoch, **raw,
                         "loss": loss.detach().item(), "gnorm": float(gn), "lr": lr_b})
            gstep += 1

            if gstep == a.calibrate_at:
                w = min(a.calibrate_window, len(hist))
                recent = {k: float(np.mean([h[k] for h in hist[-w:]])) for k in raw}
                lam = config.calibrate(recent, shares)
                print("\ncalibrated at step %d:\n%s\n"
                      % (gstep, config.describe(recent, lam)), flush=True)

            if gstep % a.log_every == 0:
                r = hist[-1]
                print("[%s] ep%d step %6d/%d | %s | gn %.2f lr %.2e | %.1f min"
                      % (a.arm, epoch, gstep, total_steps,
                         " ".join("%s %.4f" % (k, r[k]) for k in raw),
                         r["gnorm"], r["lr"], (time.time() - t0) / 60), flush=True)

            if gstep % a.save_every == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "scaler": scaler.state_dict(), "epoch": epoch,
                            "batch": bi + 1, "gstep": gstep, "hist": hist, "lam": lam},
                           ck + ".tmp")
                os.replace(ck + ".tmp", ck)     # atomic: no half checkpoint on a kill
                json.dump(hist, open(os.path.join(a.out, "history.json"), "w"))
        start_batch = 0

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "epoch": a.epochs, "batch": 0,
                "gstep": gstep, "hist": hist, "lam": lam}, ck)
    # Weights only, for downstream finetuning -- a fraction of the full checkpoint.
    torch.save({"model": model.state_dict(), "arm": a.arm, "lam": lam},
               os.path.join(a.out, "student_final.pt"))
    json.dump(hist, open(os.path.join(a.out, "history.json"), "w"))
    print("[%s] done: %d steps in %.1f min" % (a.arm, gstep, (time.time() - t0) / 60),
          flush=True)


if __name__ == "__main__":
    main()
