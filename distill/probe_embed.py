"""Embed a fixed molecule set with ONE model and write the vectors to .npy.

Deliberately one model per process. Loading the 337M teacher and a 32M student in
the same process changes the student's output even though its weights are
byte-identical (verified: same state_dict hash, cross-molecule cosine 0.5013 vs
0.5566). Their RotaryPositionalEmbeddings registers theta/cache as NON-PERSISTENT
buffers and rebuilds them lazily inside forward() when a validity check fails, and
that state does not survive two models sharing a process cleanly -- their own code
carries a comment about "environments that load this model with corrupted
non-persistent buffers", so they hit it too.

Every geometry number in this project measured before this file existed was taken
with the teacher resident and is invalid.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    # Separate arguments, not a colon-packed string: Windows paths contain a
    # drive-letter colon and any split(":") mangles them.
    ap.add_argument("--model", default=None, help="plain HF model dir (teacher)")
    ap.add_argument("--init", default=None, help="Student init dir")
    ap.add_argument("--ckpt", default=None, help="Student checkpoint, or omit for pristine init")
    ap.add_argument("--smiles-npy", required=True, help="npy of SMILES strings")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    smis = [str(x) for x in np.load(a.smiles_npy, allow_pickle=True)]

    if a.init:
        from student import Student, patch_sdpa_dtype
        patch_sdpa_dtype()
        sd = None
        if a.ckpt:
            st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
            sd = st["model"]
            print("loaded %s @ step %s" % (a.ckpt, st.get("gstep", "?")))

        # TWO CHECKPOINT LAYOUTS. The cached three-term runs trained a Student
        # wrapper, so their keys are backbone.* plus mtr_head.*. The live-KD run
        # trained a bare AutoModel and its keys have no prefix. Loading one into
        # the other raises on every key, so pick the container from the keys
        # rather than from a flag the caller has to remember to set.
        if sd is not None and not any(k.startswith("backbone.") for k in sd):
            m = AutoModel.from_pretrained(a.init, trust_remote_code=True,
                                          use_safetensors=True)
            m.load_state_dict(sd)                    # strict: any drift must raise
            fwd = lambda i, t: m(input_ids=i, attention_mask=t).mean_pool
            print("   layout: bare AutoModel (live-KD)")
        else:
            m = Student(a.init)
            if sd is not None:
                m.load_state_dict(sd)
                print("   layout: Student wrapper (cached KD)")
            fwd = lambda i, t: m(i, t)["mean_pool"]
        m = m.cuda().eval()
    else:
        m = AutoModel.from_pretrained(a.model, trust_remote_code=True,
                                      use_safetensors=True).cuda().eval()
        fwd = lambda i, t: m(input_ids=i, attention_mask=t).mean_pool

    out = []
    with torch.no_grad():
        for i in range(0, len(smis), a.batch):
            b = tok(smis[i:i + a.batch], return_tensors="pt", padding=True,
                    truncation=True, max_length=512)
            out.append(fwd(b["input_ids"].cuda(), b["attention_mask"].cuda()).float().cpu())
    Z = torch.cat(out).numpy()
    np.save(a.out, Z)
    print("wrote %s %s" % (a.out, Z.shape))


if __name__ == "__main__":
    main()
