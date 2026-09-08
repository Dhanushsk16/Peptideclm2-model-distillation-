"""Export a trained Student checkpoint as a plain HuggingFace model directory.

Their PAMPA script does AutoModel.from_pretrained(name, trust_remote_code=True),
so the student has to look like the released checkpoints: backbone weights plus
config.json / config.py / ChemPepMTR.py / tokenizer files in one flat folder.

Two things are load-bearing:

  * The MTR head is DROPPED. It exists only to carry the descriptor loss during
    distillation; their regression script builds its own head on the pooled
    embedding. Keeping ours would add keys AutoModel cannot map.

  * The folder name MUST contain "-small". resolve_model_scale() in their script
    does a substring match on the name to pick batch size (16) and learning rate
    (1e-5). A folder called "treatment" silently gets the "base" defaults --
    batch 8, lr 5e-6 -- and the arms stop being comparable.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file


def export(ckpt_path, init_dir, out_dir, tag):
    assert "-small" in os.path.basename(out_dir), \
        "folder name must contain '-small' or their script picks base-scale hyperparameters"
    os.makedirs(out_dir, exist_ok=True)

    if ckpt_path is None:                       # the untouched warm-start baseline
        shutil.copytree(init_dir, out_dir, dirs_exist_ok=True)
        print("copied pristine init -> %s" % out_dir)
    else:
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = st["model"] if "model" in st else st
        # TWO CHECKPOINT LAYOUTS.
        #   cached KD  trained a Student wrapper -> backbone.* plus mtr_head.*
        #   live KD    trained a bare AutoModel  -> no prefix at all
        # Stripping "backbone." restores the released key names in the first case;
        # in the second they are already correct and there is no head to drop.
        if any(k.startswith("backbone.") for k in sd):
            bb = {k[len("backbone."):]: v.contiguous()
                  for k, v in sd.items() if k.startswith("backbone.")}
            dropped = sorted({k.split(".")[0] for k in sd if not k.startswith("backbone.")})
        else:
            bb = {k: v.contiguous() for k, v in sd.items()}
            dropped = []
        assert bb, "empty state_dict in " + ckpt_path
        assert any(k.startswith("model.embed") for k in bb), \
            "unrecognised key layout: %s" % sorted(bb)[:3]

        for f in os.listdir(init_dir):
            if f != "model.safetensors":
                shutil.copy(os.path.join(init_dir, f), os.path.join(out_dir, f))
        save_file(bb, os.path.join(out_dir, "model.safetensors"))
        print("exported %s: %d tensors, %.1fM params, dropped %s (step %s)"
              % (tag, len(bb), sum(v.numel() for v in bb.values()) / 1e6,
                 dropped or "nothing", st.get("gstep", "?")))

    # Prove it round-trips through the same call their script will make.
    from transformers import AutoModel, AutoTokenizer
    m = AutoModel.from_pretrained(out_dir, trust_remote_code=True, use_safetensors=True)
    AutoTokenizer.from_pretrained(out_dir, trust_remote_code=True)
    n = sum(p.numel() for p in m.parameters())
    print("   verified: AutoModel loads %.1fM params from %s" % (n / 1e6, out_dir))
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="dir holding treatment/ and control/")
    ap.add_argument("--init", required=True, help="released peptideclm-2-mlm-small dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--which", default="latest.pt",
                    help="latest.pt (resumable) or student_final.pt")
    ap.add_argument("--arms", default="treatment,control",
                    help="subdirectories of --runs to export")
    ap.add_argument("--also", action="append", default=[], metavar="NAME=CKPT",
                    help="a checkpoint outside --runs, e.g. kdlive=../results/"
                         "kd_live/student_final.pt. Repeatable. NAME=CKPT rather "
                         "than NAME:CKPT because Windows paths carry a colon.")
    a = ap.parse_args()

    made = {}
    made["warmstart"] = export(None, a.init,
                               os.path.join(a.out, "peptideclm-2-mlm-small-warmstart"),
                               "warmstart")
    for arm in [x for x in a.arms.split(",") if x]:
        ck = os.path.join(a.runs, arm, a.which)
        if not os.path.exists(ck):
            print("SKIP %s: %s not found" % (arm, ck)); continue
        made[arm] = export(ck, a.init,
                           os.path.join(a.out, "peptideclm-2-mlm-small-" + arm), arm)
    for spec in a.also:
        name, _, ck = spec.partition("=")
        assert ck, "--also expects NAME=CKPT, got %r" % spec
        assert os.path.exists(ck), "checkpoint not found: " + ck
        made[name] = export(ck, a.init,
                            os.path.join(a.out, "peptideclm-2-mlm-small-" + name), name)
    json.dump(made, open(os.path.join(a.out, "exported.json"), "w"), indent=2)
    print("\n%d models exported" % len(made))


if __name__ == "__main__":
    main()
