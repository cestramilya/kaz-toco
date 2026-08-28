# Fine-tune kz-transformers/kaz-roberta-conversational for Kazakh toxic
# comment classification (head warmup, then full fine-tuning).
# Run from the repository root:
#   python models/roberta_conv/train.py
# Optional W&B logging:
#   USE_WANDB=true WANDB_ENTITY=<your-entity> python models/roberta_conv/train.py

import math
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("DATA_DIR", "./data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs/roberta_conv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv.gz")  # pandas reads gzip transparently
VAL_PATH = os.path.join(DATA_DIR, "val.csv")

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── Config ───────────────────────────────────────────────────────────────────
CFG = {
    "model_name": "kz-transformers/kaz-roberta-conversational",
    "max_length": 128,
    "batch_size": 16,
    "grad_accum": 2,  # effective batch = 32
    "lr_head": 1e-4,
    "lr_full": 1e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.10,
    "epochs_head": 1,
    "epochs_full": 3,
    "early_stop_patience": 2,
    "dropout": 0.25,
    "label_smoothing": 0.1,
    "num_labels": 2,
    "seed": SEED,
}

# ── Optional W&B logging ─────────────────────────────────────────────────────
USE_WANDB = os.environ.get("USE_WANDB", "false").lower() == "true"
if USE_WANDB:
    import wandb

    wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),  # None if not set
        project="kazakh-toxicity-classification",
        name="roberta-freeze-then-full-v2",
        config=CFG,
    )

# ── Data ─────────────────────────────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])

# log how many texts get truncated
lengths = train_df["text"].astype(str).apply(
    lambda x: len(tokenizer.encode(x, add_special_tokens=True))
)
truncated = (lengths > CFG["max_length"]).sum()
if USE_WANDB:
    wandb.log({"train/truncated_samples": int(truncated),
               "train/truncated_pct": round(truncated / len(lengths) * 100, 2)})
print(f"Truncated at max_length={CFG['max_length']}: {truncated} ({truncated/len(lengths)*100:.1f}%)")

# class balance check
print("Train balance:", train_df["label"].value_counts().to_dict())
print("Val balance:  ", val_df["label"].value_counts().to_dict())


class CommentDataset(Dataset):
    def __init__(self, df):
        self.encodings = tokenizer(
            df["text"].astype(str).tolist(),
            truncation=True, padding="max_length",
            max_length=CFG["max_length"], return_tensors="pt",
        )
        self.labels = torch.tensor(df["label"].tolist(), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


train_loader = DataLoader(CommentDataset(train_df), batch_size=CFG["batch_size"], shuffle=True)
val_loader = DataLoader(CommentDataset(val_df), batch_size=CFG["batch_size"])

# ── Model ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"  # mixed precision only on GPU
print(f"Device: {device}")

model = AutoModelForSequenceClassification.from_pretrained(
    CFG["model_name"],
    num_labels=CFG["num_labels"],
    hidden_dropout_prob=CFG["dropout"],
    attention_probs_dropout_prob=CFG["dropout"],
)
model.to(device)
if USE_WANDB:
    wandb.watch(model, log="all", log_freq=50)

# single criterion with label smoothing
criterion = torch.nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])


# ── Helpers ──────────────────────────────────────────────────────────────────
def freeze_transformer():
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_all():
    for param in model.parameters():
        param.requires_grad = True


# weight decay with parameter groups: bias and LayerNorm WITHOUT decay
def make_optimizer(lr):
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    decay_params, nodecay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n for nd in no_decay):
            nodecay_params.append(p)
        else:
            decay_params.append(p)
    grouped = [
        {"params": decay_params, "weight_decay": CFG["weight_decay"]},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return AdamW(grouped, lr=lr)


def eval_epoch(loader):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
            all_preds.extend(outputs.logits.argmax(-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, f1


def train_epoch(loader, optimizer, scheduler, scaler, phase):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast(device.type, enabled=use_amp):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels) / CFG["grad_accum"]

        scaler.scale(loss).backward()

        if (step + 1) % CFG["grad_accum"] == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            if USE_WANDB:
                wandb.log({
                    f"{phase}/grad_norm": grad_norm.item(),
                    f"{phase}/lr": scheduler.get_last_lr()[0],
                })
        total_loss += loss.item() * CFG["grad_accum"]
    return total_loss / len(loader)


def save_checkpoint(name):
    path = os.path.join(OUTPUT_DIR, name)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Checkpoint saved: {path}")


# ── Step A: head warmup (1 epoch, transformer frozen) ────────────────────────
print("\n=== Head warmup ===")
freeze_transformer()
optimizer = make_optimizer(CFG["lr_head"])
total_steps = math.ceil(len(train_loader) / CFG["grad_accum"]) * CFG["epochs_head"]
scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
scaler = GradScaler(device.type, enabled=use_amp)

train_loss = train_epoch(train_loader, optimizer, scheduler, scaler, "head")
val_loss, val_f1 = eval_epoch(val_loader)
if USE_WANDB:
    wandb.log({"head/train_loss": train_loss, "head/val_loss": val_loss, "head/val_f1": val_f1})
print(f"Head warmup — train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | val_f1: {val_f1:.4f}")

# ── Step B: full fine-tuning (early stop / model selection by val_f1) ────────
print("\n=== Full fine-tuning ===")
unfreeze_all()
optimizer = make_optimizer(CFG["lr_full"])
total_steps = math.ceil(len(train_loader) / CFG["grad_accum"]) * CFG["epochs_full"]
warmup_steps = int(total_steps * CFG["warmup_ratio"])
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
scaler = GradScaler(device.type, enabled=use_amp)

best_val_f1 = -1.0  # model selection by F1, not by loss
patience_counter = 0

for epoch in range(CFG["epochs_full"]):
    train_loss = train_epoch(train_loader, optimizer, scheduler, scaler, "full")
    val_loss, val_f1 = eval_epoch(val_loader)

    if USE_WANDB:
        wandb.log({
            "full/epoch": epoch + 1,
            "full/train_loss": train_loss,
            "full/val_loss": val_loss,
            "full/val_f1": val_f1,
        })
    print(f"Epoch {epoch+1} — train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | val_f1: {val_f1:.4f}")

    if val_f1 > best_val_f1:  # maximize F1
        best_val_f1 = val_f1
        patience_counter = 0
        save_checkpoint("best_model")
        print(f"  -> best model saved (val_f1={val_f1:.4f})")
    else:
        patience_counter += 1
        print(f"  -> patience {patience_counter}/{CFG['early_stop_patience']}")
        if patience_counter >= CFG["early_stop_patience"]:
            if USE_WANDB:
                wandb.log({"full/early_stop_epoch": epoch + 1})
            print("Early stopping triggered.")
            break

if USE_WANDB:
    wandb.log({"full/best_val_f1": best_val_f1})
    wandb.finish()
print(f"\nTraining finished. Best val_f1: {best_val_f1:.4f}")
print("Run models/roberta_conv/evaluate.py for final test metrics.")
