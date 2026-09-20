"""Write a depth-reduced copy of the model as an ordinary HuggingFace directory.

Everything downstream -- their classification script, their regression ensemble,
the PepMSND KAN pipeline -- does AutoModel.from_pretrained(path, trust_remote_code
=True). So the cheapest way to test a pruned model is to make it look like a
normal checkpoint: drop the blocks, renumber what is left, write the new
num_blocks into config.json, copy the tokenizer files across. No script changes
anywhere.

Two selection modes, for the two steps of the plan:

  --depth k        keep blocks 0..k-1. Plain truncation: the last blocks serve the
                   MLM objective, which downstream discards.
  --keep / --drop  an explicit set. This is how a Block-Influence ranking gets
                   applied -- score the blocks, then pass the survivors here.

RENUMBERING IS THE WHOLE TRICK. State-dict keys are
model.transformer.blocks.<i>.*, and nn.ModuleList indexes positionally, so after
dropping block 5 the old block 6 must be rewritten as block 5. Miss it and
load_state_dict either errors (best case) or silently leaves blocks randomly
initialised, which looks like "pruning hurt a lot" rather than a bug.

The export is verified by loading it back and comparing its pooled output against
the original model with the same blocks disabled at runtime. They must agree to
floating-point noise; anything larger means the surgery was wrong.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil

import torch
from safetensors.torch import load_file, save_file

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BLOCK_RE = re.compile(r"^(model\.transformer\.blocks\.)(\d+)(\..*)$")


def parse_set(spec, n):
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    bad = [i for i in out if not 0 <= i < n]
    assert not bad, "block index out of range for a %d-block model: %s" % (n, bad)
    return out


def export(src_dir, out_dir, keep):
    sd = load_file(os.path.join(src_dir, "model.safetensors"))
    cfg = json.load(open(os.path.join(src_dir, "config.json")))
    n = cfg["num_blocks"]
    keep = sorted(keep)
    remap = {old: new for new, old in enumerate(keep)}

    new_sd, dropped = {}, 0
    for k, v in sd.items():
        m = BLOCK_RE.match(k)
        if not m:
            new_sd[k] = v.contiguous()            # embed, final norm, sequence_head
            continue
        i = int(m.group(2))
        if i not in remap:
            dropped += 1
            continue
        new_sd[m.group(1) + str(remap[i]) + m.group(3)] = v.contiguous()

    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(src_dir):
        if f not in ("model.safetensors", "config.json") and os.path.isfile(
                os.path.join(src_dir, f)):
            shutil.copy(os.path.join(src_dir, f), os.path.join(out_dir, f))
    cfg["num_blocks"] = len(keep)
    cfg["pruned_from"] = {"source_blocks": n, "kept": keep}
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    save_file(new_sd, os.path.join(out_dir, "model.safetensors"))

    tot = sum(v.numel() for v in new_sd.values())
    print("kept %d/%d blocks -> %s" % (len(keep), n, out_dir))
    print("   %d tensors dropped | %.1fM params" % (dropped, tot / 1e6))
    return tot


def verify_weights(src_dir, out_dir, keep):
    """Prove the renumbering by comparing tensors, with no forward pass involved.

    Exported block j must be bit-identical to source block keep[j]. This is the
    decisive test for the renumbering bug, and unlike the forward comparison it
    cannot be confounded by rotary-buffer state, dtype or device. Run it first so
    that if the forward check later disagrees, we already know the weights are
    right and the problem is elsewhere.
    """
    src = load_file(os.path.join(src_dir, "model.safetensors"))
    new = load_file(os.path.join(out_dir, "model.safetensors"))
    keep = sorted(keep)
    bad = []
    for j, old in enumerate(keep):
        for k in src:
            m = BLOCK_RE.match(k)
            if not m or int(m.group(2)) != old:
                continue
            nk = m.group(1) + str(j) + m.group(3)
            if nk not in new:
                bad.append("missing " + nk)
            elif not torch.equal(src[k], new[nk]):
                bad.append("%s != %s" % (k, nk))
    extra = [k for k in new if BLOCK_RE.match(k)
             and int(BLOCK_RE.match(k).group(2)) >= len(keep)]
    non_block = [k for k in src if not BLOCK_RE.match(k)]
    for k in non_block:
        if k not in new or not torch.equal(src[k], new[k]):
            bad.append("non-block tensor changed: " + k)
    assert not bad and not extra, \
        "RENUMBERING BUG: %s %s" % (bad[:4], extra[:4])
    print("   weights: block j == source block keep[j] for all %d, %d shared "
          "tensors identical" % (len(keep), len(non_block)))


@torch.no_grad()
def verify(src_dir, out_dir, keep, device="cpu", n_mol=8):
    """Load the export and check it matches the original with the same blocks
    bypassed. This is the check that catches a renumbering mistake."""
    from transformers import AutoModel, AutoTokenizer

    from student import patch_sdpa_dtype
    from train_kd import refresh_rope, verify_rope
    patch_sdpa_dtype()

    tok = AutoTokenizer.from_pretrained(src_dir, trust_remote_code=True)
    import pandas as pd
    smis = list(pd.read_csv(os.path.join(R, "their_repo", "data",
                                         "CellPPD_test.csv")).smiles[:n_mol])
    b = tok(smis, return_tensors="pt", padding=True, truncation=True, max_length=512)
    b = {k: v.to(device) for k, v in b.items()}

    # BOTH MODELS ARE LOADED BEFORE EITHER RUNS, THEN BOTH GET refresh_rope().
    #
    # Two models of this family in one process corrupt each other's rotary state:
    # theta and cache are NON-PERSISTENT buffers rebuilt lazily inside forward(),
    # and the second load leaves the first holding wrong ones. Measured earlier at
    # cross-molecule cosine 0.5013 vs 0.5566 on byte-identical weights, which is
    # what train_kd.refresh_rope()/verify_rope() exist to repair.
    #
    # Running the reference forward before the second load is NOT a fix -- it just
    # makes the corruption intermittent, which is worse. So: load both, repair
    # both, verify both, then compare.
    ref = AutoModel.from_pretrained(src_dir, trust_remote_code=True,
                                    use_safetensors=True).to(device).eval()
    new = AutoModel.from_pretrained(out_dir, trust_remote_code=True,
                                    use_safetensors=True).to(device).eval()
    for m, nm in ((ref, "original"), (new, "export")):
        refresh_rope(m, device)
        verify_rope(m, nm)

    # Bypass the dropped blocks in the ORIGINAL by replacing them with identity,
    # which is what removing them is supposed to mean.
    blocks = ref.model.transformer.blocks
    keep = set(keep)

    class Identity(torch.nn.Module):
        def forward(self, x, input_pos=None, mask=None):
            return x

    for i in range(len(blocks)):
        if i not in keep:
            blocks[i] = Identity()

    a = ref(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
    a = (a["mean_pool"] if isinstance(a, dict) else a.mean_pool).float()
    c = new(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
    c = (c["mean_pool"] if isinstance(c, dict) else c.mean_pool).float()

    err = (a - c).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(a, c, dim=1).min().item()
    ok = err < 1e-3
    print("   forward: max|diff| %.2e  min cosine %.6f  ->  %s"
          % (err, cos, "OK" if ok else "MISMATCH"))
    assert ok, ("export does not reproduce the bypassed original. The weight check "
                "above already passed, so this is NOT a renumbering bug -- suspect "
                "rotary buffer state from two models sharing a process.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="model dir; default = 337M teacher")
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=None, help="keep blocks 0..depth-1")
    ap.add_argument("--keep", default=None, help="explicit block list, e.g. 0-9,12,15")
    ap.add_argument("--drop", default=None, help="explicit blocks to remove")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()

    src = a.src or glob.glob(
        R + "/models/models--aaronfeller--peptideclm-2-mlm-large/snapshots/*")[0]
    n = json.load(open(os.path.join(src, "config.json")))["num_blocks"]

    given = [x is not None for x in (a.depth, a.keep, a.drop)]
    assert sum(given) == 1, "give exactly one of --depth, --keep, --drop"
    if a.depth is not None:
        assert 0 < a.depth <= n, "--depth must be in 1..%d" % n
        keep = set(range(a.depth))
    elif a.keep is not None:
        keep = parse_set(a.keep, n)
    else:
        keep = set(range(n)) - parse_set(a.drop, n)
    assert keep, "nothing left to keep"

    export(src, a.out, keep)
    if not a.no_verify:
        verify_weights(src, a.out, keep)     # cheap, decisive, no forward pass
        verify(src, a.out, keep, a.device)


if __name__ == "__main__":
    main()
