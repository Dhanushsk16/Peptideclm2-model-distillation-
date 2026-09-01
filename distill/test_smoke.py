"""Correctness tests for the distillation pipeline. Run before any real training.

Deliberately checks properties, not just "does it run": a loss that returns a
number is not evidence it returns the right number, and a pipeline can pass every
shape assertion while training against the wrong rows.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import config
from data import ShardStream, SubsetTable, build_batches
from losses import kd_loss, mtr_loss, spkd_loss
from student import Student

OK, FAIL = "  ok  ", " FAIL "
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print("[%s] %-48s %s" % (OK if cond else FAIL, name, detail))


# ---------------------------------------------------------------- losses
def test_losses():
    print("\n=== loss unit tests ===")
    g = torch.Generator().manual_seed(0)
    N, V, K = 256, 405, 16

    tl = torch.randn(N, V, generator=g) * 3
    topv_full, topi = torch.softmax(tl, -1).log().topk(K, -1)
    topv = topv_full.half().float()          # round-trip through fp16, like the cache

    l_same = kd_loss(tl, topi, topv)
    check("KD ~ 0 when student == teacher", l_same.item() < 1e-3, "%.3e" % l_same.item())

    l_rand = kd_loss(torch.randn(N, V, generator=g), topi, topv)
    l_unif = kd_loss(torch.zeros(N, V), topi, topv)
    check("KD > 0 for a wrong student", l_rand.item() > 0, "%.4f" % l_rand.item())
    check("KD(random) > KD(exact)", l_rand.item() > l_same.item(), "%.4f" % l_rand.item())
    check("KD(uniform) > KD(exact)", l_unif.item() > l_same.item(), "%.4f" % l_unif.item())

    # The residual bucket must survive top-k mass rounding to exactly 1.
    tv = torch.full((8, K), float(np.log(1.0 / K)))
    l_edge = kd_loss(torch.randn(8, V, generator=g), torch.arange(K).repeat(8, 1), tv)
    check("KD finite when top-k mass == 1", bool(torch.isfinite(l_edge).all()),
          "%.4f" % l_edge.item())

    check("MTR clips extreme targets",
          abs(mtr_loss(torch.zeros(4, 99), torch.full((4, 99), 583.0)).item() - 100.0) < 1e-3,
          "loss == 10^2")

    z = torch.randn(32, 64, generator=g)
    check("SPKD == 0 for identical pools", spkd_loss(z, z).item() < 1e-12)
    q, _ = torch.linalg.qr(torch.randn(64, 64, generator=g))
    check("SPKD invariant to rotation", spkd_loss(z @ q, z).item() < 1e-10,
          "%.3e" % spkd_loss(z @ q, z).item())
    check("SPKD handles width mismatch (512 vs 1024)",
          np.isfinite(spkd_loss(torch.randn(32, 512), torch.randn(32, 1024)).item()))
    check("SPKD > 0 for unrelated pools",
          spkd_loss(z, torch.randn(32, 64, generator=g)).item() > 0)


def test_batching():
    print("\n=== batching ===")
    lengths = np.concatenate([np.full(200, 23), np.full(200, 340)])   # bimodal, as the corpus is
    batches = build_batches(lengths, max_tokens=4096)
    check("every batch within the token budget",
          all(lengths[b].max() * len(b) <= 4096 for b in batches))
    check("all molecules appear exactly once",
          sorted(i for b in batches for i in b) == list(range(len(lengths))))
    pad = [1 - lengths[b].sum() / (lengths[b].max() * len(b)) for b in batches]
    check("length bucketing keeps padding low", float(np.mean(pad)) < 0.05,
          "mean padding %.2f%%" % (100 * float(np.mean(pad))))


# ---------------------------------------------------------------- data
def test_data(stream, table, tok):
    print("\n=== data tests ===")
    b = next(iter(stream.iter_epoch(0)))
    T = b["input_ids"].shape[1]

    check("masked positions inside sequence", int(b["cols"].max()) < T,
          "max col %d < T %d" % (int(b["cols"].max()), T))
    check("[CLS] never masked", int(b["cols"].min()) >= 1)
    check("no masked position lands on padding",
          bool((b["attention_mask"][b["rows"], b["cols"]] == 1).all()))
    check("MASK token written at every position",
          bool((b["input_ids"][b["rows"], b["cols"]] == stream.MASK).all()))
    check("labels captured BEFORE masking (never [MASK])",
          bool((b["labels"] != stream.MASK).all()))

    check("targets aligned with positions",
          b["topi"].shape[0] == b["cols"].shape[0] == b["topv"].shape[0],
          "%d positions" % b["cols"].shape[0])
    check("teacher pool is 1024-d", b["teacher_pool"].shape[1] == 1024)
    check("descriptors are 99-d", b["desc"].shape[1] == 99)

    p = b["topv"].exp().sum(-1)
    check("teacher top-16 mass <= 1", float(p.max()) <= 1.001, "max %.6f" % float(p.max()))
    check("teacher mass is high", float(p.mean()) > 0.9, "mean %.4f" % float(p.mean()))

    real = int(b["attention_mask"].sum()) - 2 * b["input_ids"].shape[0]
    check("masking rate ~ 0.25", 0.20 < b["cols"].shape[0] / real < 0.28,
          "%.4f" % (b["cols"].shape[0] / real))

    # Mask A and B must be disjoint on the same molecules.
    sb = ShardStream(os.path.dirname(stream.paths[0]), table, tok, mask="b",
                     max_tokens=stream.max_tokens, seed=stream.seed)
    sb.paths = list(stream.paths)
    bb = next(iter(sb.iter_epoch(0)))
    ov = len(set(zip(b["rows"].tolist(), b["cols"].tolist())) &
             set(zip(bb["rows"].tolist(), bb["cols"].tolist())))
    check("mask A and B disjoint", ov == 0, "%d overlapping (row,col)" % ov)


# ---------------------------------------------------------------- fwd/bwd
def test_fwd_bwd(model, stream, device):
    print("\n=== forward / backward on %s ===" % device)
    model.to(device).train()
    opt = torch.optim.AdamW(model.param_groups(config.LR_BACKBONE, config.LR_HEAD),
                            betas=config.BETAS)
    scaler = torch.amp.GradScaler(device.split(":")[0], enabled=(device != "cpu"))
    lam = config.LAMBDAS

    hist, gn = [], None
    t0 = time.time()
    for step, batch in enumerate(stream.iter_epoch(0)):
        if step >= 3:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device.split(":")[0], dtype=torch.float16,
                            enabled=(device != "cpu")):
            out = model(batch["input_ids"], batch["attention_mask"])
            sel = out["logits"][batch["rows"], batch["cols"]]
            parts = {
                "kd": kd_loss(sel, batch["topi"], batch["topv"]),
                "mtr": mtr_loss(out["mtr"], batch["desc"], clip=config.MTR_CLIP),
                "sp": spkd_loss(out["mean_pool"], batch["teacher_pool"]),
                "mlm": F.cross_entropy(sel.float(), batch["labels"]),
            }
            loss = sum(lam[k] * parts[k] for k in ("kd", "mtr", "sp"))

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        scaler.step(opt)
        scaler.update()

        hist.append({k: v.detach().item() for k, v in parts.items()})
        print("   step %d  B=%3d T=%3d pos=%5d | kd %.4f mtr %.4f sp %.6f mlm %.4f | gn %.2f"
              % (step, batch["input_ids"].shape[0], batch["input_ids"].shape[1],
                 batch["cols"].shape[0], hist[-1]["kd"], hist[-1]["mtr"],
                 hist[-1]["sp"], hist[-1]["mlm"], gn))

    check("forward/backward completes", len(hist) == 3, "%.1fs" % (time.time() - t0))
    check("all losses finite", all(np.isfinite(list(h.values())).all() for h in hist))
    check("gradient norm finite and > 0", bool(torch.isfinite(gn)) and gn > 0, "%.3f" % gn)

    nograd = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    check("every parameter received a gradient", not nograd, "%d missing" % len(nograd))
    bb = sum(p.grad.abs().sum().item() for p in model.backbone.parameters()
             if p.grad is not None)
    check("backbone gradients non-zero", bb > 0, "sum|g| %.3e" % bb)

    # SPKD only touches mean_pool -- confirm it still reaches the whole encoder.
    model.zero_grad(set_to_none=True)
    batch = {k: v.to(device) for k, v in next(iter(stream.iter_epoch(0))).items()}
    out = model(batch["input_ids"], batch["attention_mask"])
    spkd_loss(out["mean_pool"], batch["teacher_pool"]).backward()
    ge = model.backbone.model.embed.weight.grad
    check("SPKD alone reaches the embedding", ge is not None and ge.abs().sum() > 0,
          "%.3e" % (ge.abs().sum().item() if ge is not None else 0))

    if device != "cpu":
        print("   peak GPU memory: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))


def test_overfit(model, stream, device, steps=60):
    """Strongest check available: on ONE batch the loss must fall a long way. If
    it does not, inputs and targets are misaligned somewhere."""
    print("\n=== overfit one batch (%d steps) ===" % steps)
    batch = {k: v.to(device) for k, v in next(iter(stream.iter_epoch(0))).items()}
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    first = last = None
    for s in range(steps):
        out = model(batch["input_ids"], batch["attention_mask"])
        sel = out["logits"][batch["rows"], batch["cols"]]
        l = kd_loss(sel, batch["topi"], batch["topv"])
        opt.zero_grad(set_to_none=True)
        l.backward()
        opt.step()
        first = l.item() if s == 0 else first
        last = l.item()
        if s % 20 == 0 or s == steps - 1:
            print("   step %3d  kd %.4f" % (s, l.item()))
    check("KD loss drops on a single batch", last < first * 0.5,
          "%.4f -> %.4f (%.0f%% down)" % (first, last, 100 * (1 - last / first)))


# ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--shard-limit", type=int, default=1)
    a = ap.parse_args()

    if a.device != "cpu":
        p = torch.cuda.get_device_properties(0)
        print("device: %s  %.1f GB" % (p.name, p.total_memory / 1e9))

    test_losses()
    test_batching()

    tok = AutoTokenizer.from_pretrained(a.student, trust_remote_code=True)
    table = SubsetTable(a.subset)
    stream = ShardStream(a.cache, table, tok, mask="a", max_tokens=a.max_tokens,
                         seed=0, shard_limit=a.shard_limit)
    print("\nsubset %d molecules | %d shard(s) under test"
          % (len(table.smiles), len(stream.paths)))

    test_data(stream, table, tok)

    model = Student(a.student)
    print("\nstudent: %.1fM params (backbone %.1fM + mtr head %.1fM)"
          % (sum(p.numel() for p in model.parameters()) / 1e6,
             sum(p.numel() for p in model.backbone.parameters()) / 1e6,
             sum(p.numel() for p in model.mtr_head.parameters()) / 1e6))

    test_fwd_bwd(model, stream, a.device)
    test_overfit(model, stream, a.device)

    print("\n%d/%d checks passed" % (sum(results), len(results)))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
