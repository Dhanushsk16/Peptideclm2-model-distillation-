"""PAMPA regression with LoRA instead of full finetuning.

Drop-in replacement for their finetune_ensemble.py worker: it accepts the same
--worker_test_fold / --worker_val_fold / --worker_output_path arguments and writes
the same row_idx/prediction CSV, so run_pampa.py drives it unchanged:

    python run_pampa.py --script pampa_lora.py ...

EVERYTHING except the trainable-parameter set is copied from their regression
script, so the two runs differ in one variable only:

    head         Linear(d,d) -> SiLU -> Dropout -> Linear(d,1)     (theirs)
    loss         MSELoss                                           (theirs)
    optimizer    AdamW, weight_decay 1e-3                          (theirs)
    schedule     linear warmup 10% -> linear decay                 (theirs)
    batch size   16 for "-small"                                   (theirs)
    early stop   val_rmse, patience 20, max 250 epochs             (theirs)
    folds        cluster -> fold, nested CV, ensemble inner folds   (theirs)
    LoRA         r=16, alpha=32, dropout 0.1, target qkv_proj       (their CLASSIFICATION config)

The LoRA config is lifted verbatim from their classification_finetuning_v2.py,
which is what they actually used for CellPPD/AmpHGT/THPep. So this run answers:
does the distilled backbone help when the backbone is FROZEN and only ~0.9% of
weights adapt?

That is the regime where a better representation should matter MOST. Full
finetuning can reshape a mediocre representation into a good one given enough
capacity and steps; LoRA cannot. If distillation improved the frozen features,
this is where it should show.

LEARNING RATE: 3e-4, not their full-finetune 1e-5. LoRA adapters start at zero and
must move much further than a pretrained weight; 1e-5 would barely train them.
3e-4 is the value their own LoRA classification script uses.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from student import patch_sdpa_dtype

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------- data
def normalize_frame(frame):
    """Same normalisation their script applies: PAMPA -> value, cluster -> fold,
    then sort by (fold, SMILES). The sort order is what makes row_idx in the
    output line up with the driver's expectation."""
    f = frame.copy()
    if "value" not in f.columns:
        f = f.rename(columns={"PAMPA": "value"})
    if "fold" not in f.columns:
        clusters = sorted(f["cluster"].dropna().unique().tolist())
        f["fold"] = f["cluster"].map({c: i for i, c in enumerate(clusters)})
    if "SMILES" not in f.columns:
        f = f.rename(columns={"smiles": "SMILES"})
    cols = [c for c in ["SMILES", "value", "fold", "cluster"] if c in f.columns]
    f = f[cols].copy()
    f["value"] = f["value"].astype(np.float32)
    f["fold"] = f["fold"].astype(int)
    return f.sort_values(["fold", "SMILES"]).reset_index(drop=True)


class RegressionDataset(Dataset):
    def __init__(self, smiles, labels, tokenizer, max_length=2048):
        enc = tokenizer(list(smiles), truncation=True, padding=False,
                        max_length=max_length, add_special_tokens=True)
        self.ids = [torch.tensor(x, dtype=torch.long) for x in enc["input_ids"]]
        self.labels = torch.tensor(list(labels), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "label": self.labels[i]}


def make_collate(pad_id):
    def collate(batch):
        ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True,
                           padding_value=pad_id)
        return {"input_ids": ids,
                "attention_mask": (ids != pad_id).long(),
                "labels": torch.stack([b["label"] for b in batch])}
    return collate


# ---------------------------------------------------------------- model
class LoRARegressionModel(pl.LightningModule):
    def __init__(self, model_name, learning_rate, total_steps, head_dropout,
                 weight_decay, warmup_fraction, lora_r, lora_alpha, lora_dropout):
        super().__init__()
        self.save_hyperparameters()
        patch_sdpa_dtype()

        base = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        cfg = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=lora_r,
                         lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                         target_modules=["qkv_proj"])
        self.model = get_peft_model(base, cfg)
        d = base.config.embed_dim
        # Identical head to their full-finetune script.
        self.regression_head = nn.Sequential(
            nn.Linear(d, d), nn.SiLU(), nn.Dropout(head_dropout), nn.Linear(d, 1))
        self.criterion = nn.MSELoss()

        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.parameters())
        print("trainable %.2fM / %.1fM (%.2f%%)" % (tr / 1e6, tot / 1e6, 100 * tr / tot),
              flush=True)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out["mean_pool"] if isinstance(out, dict) else out.mean_pool
        return self.regression_head(pooled).squeeze(-1)

    def training_step(self, batch, _):
        loss = self.criterion(self(batch["input_ids"], batch["attention_mask"]),
                              batch["labels"])
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        mse = self.criterion(self(batch["input_ids"], batch["attention_mask"]),
                             batch["labels"])
        self.log("val_loss", mse, on_step=False, on_epoch=True)
        self.log("val_rmse", torch.sqrt(mse + 1e-12), prog_bar=True,
                 on_step=False, on_epoch=True)
        return mse

    def predict_step(self, batch, _):
        return self(batch["input_ids"], batch["attention_mask"])

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate,
                                weight_decay=self.hparams.weight_decay)
        sch = get_linear_schedule_with_warmup(
            opt,
            num_warmup_steps=max(1, int(self.hparams.warmup_fraction * self.hparams.total_steps)),
            num_training_steps=max(1, self.hparams.total_steps))
        return [opt], [{"scheduler": sch, "interval": "step"}]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--gpu_index", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_epochs", type=int, default=250)
    # Their default. Anything below one epoch stops training mid-epoch, so the
    # epoch-end validation never runs and EarlyStopping raises on a missing
    # val_rmse. Learned the hard way on the full-finetune run.
    ap.add_argument("--max_steps", type=int, default=10000)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--learning_rate", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--head_dropout", type=float, default=0.1)
    ap.add_argument("--warmup_fraction", type=float, default=0.10)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--log_dir", default="/tmp/pampa_lora_logs")
    ap.add_argument("--worker_test_fold", type=int, required=True)
    ap.add_argument("--worker_val_fold", type=int, required=True)
    ap.add_argument("--worker_output_path", required=True)
    a = ap.parse_args()

    pl.seed_everything(a.seed, workers=True)
    frame = normalize_frame(pd.read_csv(a.data_csv))
    test_df = frame[frame.fold == a.worker_test_fold].reset_index(drop=True)
    rest = frame[frame.fold != a.worker_test_fold]
    val_df = rest[rest.fold == a.worker_val_fold].reset_index(drop=True)
    train_df = rest[rest.fold != a.worker_val_fold].reset_index(drop=True)
    print("[lora] test_fold=%d val_fold=%d train_n=%d val_n=%d test_n=%d"
          % (a.worker_test_fold, a.worker_val_fold, len(train_df), len(val_df), len(test_df)),
          flush=True)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    coll = make_collate(tok.pad_token_id)
    mk = lambda df: RegressionDataset(df.SMILES, df.value, tok, a.max_length)
    tl = DataLoader(mk(train_df), batch_size=a.batch_size, shuffle=True,
                    collate_fn=coll, num_workers=0)
    vl = DataLoader(mk(val_df), batch_size=64, collate_fn=coll, num_workers=0)
    pl_ = DataLoader(mk(test_df), batch_size=64, collate_fn=coll, num_workers=0)

    steps = max(1, len(tl)) * a.max_epochs
    model = LoRARegressionModel(a.model, a.learning_rate, min(steps, a.max_steps),
                                a.head_dropout, a.weight_decay, a.warmup_fraction,
                                a.lora_r, a.lora_alpha, a.lora_dropout)

    run = "f%d_v%d" % (a.worker_test_fold, a.worker_val_fold)
    ckpt = ModelCheckpoint(monitor="val_rmse", mode="min", save_top_k=1, filename="best")
    trainer = pl.Trainer(
        accelerator="cuda", devices=[a.gpu_index],
        max_epochs=a.max_epochs, max_steps=a.max_steps,
        callbacks=[ckpt, EarlyStopping(monitor="val_rmse", mode="min",
                                       patience=a.patience, min_delta=1e-4)],
        logger=CSVLogger(a.log_dir, name=run),
        log_every_n_steps=10, check_val_every_n_epoch=1,
        enable_progress_bar=False, gradient_clip_val=1.0)
    trainer.fit(model, train_dataloaders=tl, val_dataloaders=vl)

    preds = trainer.predict(model, dataloaders=pl_, ckpt_path=ckpt.best_model_path)
    flat = np.concatenate([p.detach().float().cpu().view(-1).numpy() for p in preds])
    os.makedirs(os.path.dirname(a.worker_output_path) or ".", exist_ok=True)
    pd.DataFrame({"row_idx": np.arange(len(flat), dtype=np.int64),
                  "prediction": flat.astype(np.float32)}).to_csv(a.worker_output_path,
                                                                 index=False)
    print("[lora] wrote %s (%d rows)" % (a.worker_output_path, len(flat)), flush=True)


if __name__ == "__main__":
    main()
