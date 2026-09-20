"""How deep does the encoder actually need to be for downstream use?

The model is used downstream as: mean-pooled vector -> small trained head. The MLM
head is discarded. So the last blocks may be doing work that only serves the
pretraining objective, and truncating the stack could be free. This measures that
directly, before any pruning algorithm is written.

WHY THIS COMES FIRST. It is the ceiling on every depth-pruning method. No cleverer
criterion can beat simply cutting at the depth where the representation saturates,
so if the curve is flat from layer k upward you already know the answer, and if it
is not flat there is no slack for a smarter method to find either.

WHAT IS MEASURED. A logistic-regression probe on FROZEN pooled embeddings from each
block, with the L2 strength cross-validated on train and MCC reported on the held-out
test set. That is a lower bound on what the real head achieves -- their head is an
MLP with LoRA adapters in the backbone -- but it is the right shape of curve and it
costs minutes instead of GPU-hours.

TWO POOLING VARIANTS, because the difference matters:

  post-norm   block k's output passed through the stack's FINAL LayerNorm, then
              pooled. This is what an actually-truncated model would compute, since
              truncation keeps that norm. THIS IS THE DEPLOYABLE NUMBER.
  raw         block k's output pooled directly. Diagnostic only: residual-stream
              magnitude grows with depth, so raw vectors are not comparable across
              layers, but a gap between the two columns says the final LayerNorm's
              learned gain/bias is mistuned for that depth.

Pooling replicates MLM_core.forward exactly -- masked on (ids != 0), which includes
[CLS] and [SEP] -- so the deepest layer's post-norm vector must equal the model's own
mean_pool. That equality is asserted rather than assumed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from student import patch_sdpa_dtype

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

BENCH = {
    # name: (train csv, test csv, smiles column, label column)
    "CellPPD": ("their_repo/data/CellPPD_train.csv", "their_repo/data/CellPPD_test.csv",
                "smiles", "label"),
    "AmpHGT": ("their_repo/data/amp_train.csv", "their_repo/data/amp_test.csv",
               "smiles", "label"),
}


def load_bench(name):
    tr, te, sc, lc = BENCH[name]
    a = pd.read_csv(os.path.join(R, tr))
    b = pd.read_csv(os.path.join(R, te))
    return (list(a[sc]), a[lc].to_numpy().astype(int),
            list(b[sc]), b[lc].to_numpy().astype(int))


@torch.no_grad()
def pooled_per_layer(model, tok, smiles, device, batch=16, max_length=512):
    """Return (n_layers+1, N, d) pooled embeddings: index 0 = embedding output,
    index k = after block k. Both raw and post-final-norm."""
    core = model.model                       # MLM_core
    blocks = core.transformer.blocks
    final_norm = core.transformer.norm

    # Pool INSIDE the hook. Collecting all 33 hidden states and converting them to
    # fp32 afterwards costs 33 x B x T x d x 4 bytes at once -- ~1 GB per batch at
    # B=16, T=480 -- which on a 4 GB card means the allocator thrashes for the
    # whole run. Pooling on arrival keeps one (B, T, d) tensor live instead of 33.
    st = {"m": None, "denom": None, "raw": [], "post": []}

    def hook(mod, inp, out):
        o = out.float()
        st["raw"].append(((o * st["m"]).sum(1) / st["denom"]).cpu())
        n = final_norm(out.to(final_norm.weight.dtype)).float()
        st["post"].append(((n * st["m"]).sum(1) / st["denom"]).cpu())

    hooks = [b.register_forward_hook(hook) for b in blocks]

    raw, post, ref = [], [], []
    for i in range(0, len(smiles), batch):
        chunk = smiles[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length)
        ids = enc["input_ids"].to(device)
        att = enc["attention_mask"].to(device)

        # Exactly MLM_core.forward: mask on (ids != pad), pad hardcoded to 0 there.
        st["m"] = (ids != 0).unsqueeze(-1).float()
        st["denom"] = st["m"].sum(1).clamp(min=1.0)
        st["raw"].clear(); st["post"].clear()

        e = core.embed(ids).float()
        emb_raw = ((e * st["m"]).sum(1) / st["denom"]).cpu()
        emb_post = ((final_norm(e.to(final_norm.weight.dtype)).float() * st["m"]).sum(1)
                    / st["denom"]).cpu()
        del e

        with torch.autocast(device.split(":")[0], dtype=torch.float16,
                            enabled=(device != "cpu")):
            out = model(input_ids=ids, attention_mask=att)

        raw.append(torch.stack([emb_raw] + list(st["raw"])))
        post.append(torch.stack([emb_post] + list(st["post"])))
        ref.append((out["mean_pool"] if isinstance(out, dict) else out.mean_pool).float().cpu())
        if i and (i // batch) % 50 == 0:
            print("   %d/%d" % (i, len(smiles)), flush=True)

    for h in hooks:
        h.remove()
    raw = torch.cat(raw, dim=1).numpy()
    post = torch.cat(post, dim=1).numpy()
    ref = torch.cat(ref).numpy()

    # The deepest post-norm pool IS the model's own mean_pool. If this fails the
    # hook is capturing the wrong tensor and every number below is meaningless.
    err = np.abs(post[-1] - ref).max()
    assert err < 2e-2, "pooling does not reproduce model mean_pool (max err %.3g)" % err
    print("   pooling verified against model mean_pool (max err %.2e)" % err)
    return raw, post


def boot_ci(y, pred, prob, n=1000, seed=0):
    """Bootstrap 95% CI on test MCC. Without this the depth curve is unreadable:
    on a 300-row test set the sampling noise is comparable to the whole spread
    across depths, and picking the argmax depth is then fitting noise."""
    rng = np.random.default_rng(seed)
    m, a = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        m.append(matthews_corrcoef(y[i], pred[i]))
        a.append(roc_auc_score(y[i], prob[i]))
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)),
            float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))


def probe(Xtr, ytr, Xte, yte, seed=0, ci=False, cs=(0.001, 0.01, 0.1, 1.0, 10.0)):
    """L2 logistic regression, C chosen by 5-fold CV on train only."""
    best, best_s = None, -2
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for C in cs:
        s = []
        for tr, va in skf.split(Xtr, ytr):
            p = make_pipeline(StandardScaler(),
                              LogisticRegression(C=C, max_iter=2000))
            p.fit(Xtr[tr], ytr[tr])
            s.append(matthews_corrcoef(ytr[va], p.predict(Xtr[va])))
        if np.mean(s) > best_s:
            best, best_s = C, np.mean(s)
    p = make_pipeline(StandardScaler(), LogisticRegression(C=best, max_iter=2000))
    p.fit(Xtr, ytr)
    pred = p.predict(Xte)
    prob = p.predict_proba(Xte)[:, 1]
    out = dict(C=best, cv_mcc=best_s, mcc=matthews_corrcoef(yte, pred),
               auc=roc_auc_score(yte, prob), acc=accuracy_score(yte, pred))
    if ci:
        lo, hi, alo, ahi = boot_ci(yte, pred, prob, seed=seed)
        out.update(mcc_lo=lo, mcc_hi=hi, auc_lo=alo, auc_hi=ahi)
    return out


def bag_of_tokens(tok, smiles, vocab_size, max_length=512):
    """Token-count vector per molecule -- no transformer at all.

    The control that decides whether the depth curve means anything. If counting
    the 405 vocabulary items scores as well as the 337M encoder, the benchmark is
    a composition task, the encoder's contextual structure is not what is being
    measured, and no conclusion about representation quality can be drawn from it.
    """
    X = np.zeros((len(smiles), vocab_size), dtype=np.float32)
    for i, s in enumerate(smiles):
        for t in tok(s, add_special_tokens=False, truncation=True,
                     max_length=max_length)["input_ids"]:
            X[i, t] += 1
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="model dir; default = the 337M teacher")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--bench", default="CellPPD", choices=sorted(BENCH))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--out", default=None)
    ap.add_argument("--depths", default=None,
                    help="comma-separated depths to probe instead of all 0..nb. "
                         "With a warm embedding cache this is CPU-only, which is "
                         "how the raw-vs-post-norm question gets settled without "
                         "re-extracting anything on a GPU")
    ap.add_argument("--fast", action="store_true",
                    help="3-value C grid and skip the raw-pool diagnostic; for the "
                         "large benchmarks where the probe fits dominate runtime")
    a = ap.parse_args()

    mdir = a.model or glob.glob(
        R + "/models/models--aaronfeller--peptideclm-2-mlm-large/snapshots/*")[0]
    tag = a.tag or os.path.basename(mdir.rstrip("/\\"))

    patch_sdpa_dtype()
    tok = AutoTokenizer.from_pretrained(mdir, trust_remote_code=True)
    model = AutoModel.from_pretrained(mdir, trust_remote_code=True,
                                      use_safetensors=True).to(a.device).eval()
    nb = len(model.model.transformer.blocks)
    d = model.model.embed.weight.shape[1]
    tot = sum(p.numel() for p in model.parameters())
    per_block = (tot - sum(p.numel() for n, p in model.named_parameters()
                           if "blocks" not in n)) / nb
    print("%s: %d blocks, d=%d, %.1fM params (%.2fM/block)"
          % (tag, nb, d, tot / 1e6, per_block / 1e6))

    str_tr, y_tr, str_te, y_te = load_bench(a.bench)
    print("%s: train %d (%d pos) | test %d (%d pos)"
          % (a.bench, len(y_tr), y_tr.sum(), len(y_te), y_te.sum()))

    cache = os.path.join(R, "results", "pruning", "depth_probe", "depth_emb_%s_%s.npz" % (tag[:12], a.bench))
    if os.path.exists(cache):
        z = np.load(cache)
        raw_tr, post_tr, raw_te, post_te = (z["raw_tr"], z["post_tr"],
                                            z["raw_te"], z["post_te"])
        print("loaded cached embeddings from " + cache)
    else:
        print("extracting train embeddings...")
        raw_tr, post_tr = pooled_per_layer(model, tok, str_tr, a.device, a.batch,
                                           a.max_length)
        print("extracting test embeddings...")
        raw_te, post_te = pooled_per_layer(model, tok, str_te, a.device, a.batch,
                                           a.max_length)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, raw_tr=raw_tr, post_tr=post_tr,
                            raw_te=raw_te, post_te=post_te)

    # Control first: it sets the bar every depth has to clear to mean anything.
    CS = (0.01, 0.1, 1.0) if a.fast else (0.001, 0.01, 0.1, 1.0, 10.0)
    b = probe(bag_of_tokens(tok, str_tr, model.config.vocab_size, a.max_length), y_tr,
              bag_of_tokens(tok, str_te, model.config.vocab_size, a.max_length), y_te,
              ci=True, cs=CS)
    print("\nCONTROL  bag-of-tokens (no transformer, %d counts):" % model.config.vocab_size)
    print("         MCC %.4f [%.4f, %.4f]   AUC %.4f"
          % (b["mcc"], b["mcc_lo"], b["mcc_hi"], b["auc"]))

    # A subset run exists to compare the two pooling variants at a few depths, so
    # it must NOT use --fast (which skips the raw column -- the whole point).
    depths = (sorted(int(x) for x in a.depths.split(",") if x.strip())
              if a.depths else list(range(nb + 1)))
    if a.depths:
        assert not a.fast, "--depths is for the raw-vs-post comparison; drop --fast"

    rows = []
    print("\n%-6s %8s %-18s %8s %-16s %8s"
          % ("depth", "MCC", "95% CI", "AUC", "95% CI", "params"))
    for k in depths:
        p = probe(post_tr[k], y_tr, post_te[k], y_te, ci=True, cs=CS)
        r = ({"mcc": float("nan"), "auc": float("nan")} if a.fast
             else probe(raw_tr[k], y_tr, raw_te[k], y_te, cs=CS))
        kept = (tot - per_block * (nb - k)) / 1e6
        rows.append(dict(depth=k, params_M=kept, mcc=p["mcc"], mcc_lo=p["mcc_lo"],
                         mcc_hi=p["mcc_hi"], auc=p["auc"], auc_lo=p["auc_lo"],
                         auc_hi=p["auc_hi"], acc=p["acc"], C=p["C"],
                         cv_mcc=p["cv_mcc"], mcc_raw=r["mcc"], auc_raw=r["auc"]))
        print("%-6s %8.4f [%.4f, %.4f] %8.4f [%.4f, %.4f] %6.1fM"
              % ("emb" if k == 0 else str(k), p["mcc"], p["mcc_lo"], p["mcc_hi"],
                 p["auc"], p["auc_lo"], p["auc_hi"], kept), flush=True)

    df = pd.DataFrame(rows)
    if a.depths:
        # The summary below compares against the full stack and the full spread,
        # neither of which a subset has. Print the pooling comparison instead.
        #
        # post = block k's output through the stack's FINAL LayerNorm, which is what
        # a truncated model computes. raw = pooled straight from block k. That
        # LayerNorm was trained for block-32 outputs and its per-token rescaling is
        # NOT absorbable by a linear probe, so a large delta means part of the
        # depth curve's shape is the norm mismatch rather than the representation.
        print("\n%-6s %8s %8s %9s   %8s %8s"
              % ("depth", "post", "raw", "delta", "auc_post", "auc_raw"))
        for r in rows:
            print("%-6d %8.4f %8.4f %+9.4f   %8.4f %8.4f"
                  % (r["depth"], r["mcc"], r["mcc_raw"], r["mcc"] - r["mcc_raw"],
                     r["auc"], r["auc_raw"]))
        out = a.out or os.path.join(
            R, "results", "pruning", "depth_probe", "depth_probe_%s_%s_subset.csv" % (tag, a.bench))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df.to_csv(out, index=False)
        print("\nwrote " + out)
        return
    full = df.iloc[-1]
    print("\nfull depth (%d): MCC %.4f [%.4f, %.4f] at %.1fM"
          % (full.depth, full.mcc, full.mcc_lo, full.mcc_hi, full.params_M))

    # The only defensible reading: shallowest depth whose CI overlaps the full
    # stack's point estimate. Taking the argmax over 33 noisy depths would report
    # a winner even if the true curve were perfectly flat.
    ok = df[df.mcc_hi >= full.mcc]
    if len(ok):
        s = ok.iloc[0]
        print("shallowest depth whose 95%% CI reaches full-stack MCC: %d, %.1fM (%.0f%%)"
              % (s.depth, s.params_M, 100 * s.params_M / full.params_M))
    width = (df.mcc_hi - df.mcc_lo).mean()
    print("mean CI width %.4f vs spread across depths %.4f -- %s"
          % (width, df.mcc.max() - df.mcc.min(),
             "curve is NOT resolvable, differences are noise"
             if width > (df.mcc.max() - df.mcc.min()) * 0.7 else "curve is resolvable"))
    print("AUC is the steadier readout (threshold-free): peak %.4f at depth %d, "
          "full %.4f" % (df.auc.max(), int(df.loc[df.auc.idxmax(), "depth"]), full.auc))

    out = a.out or os.path.join(R, "results", "pruning", "depth_probe", "depth_probe_%s_%s.csv" % (tag, a.bench))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print("\nwrote " + out)


if __name__ == "__main__":
    main()
