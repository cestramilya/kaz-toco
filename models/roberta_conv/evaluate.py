# Final evaluation of the fine-tuned kaz-roberta-conversational model on the
# held-out test set (comments from 5 videos never seen during training).
# Run from the repository root, after models/roberta_conv/train.py:
#   python models/roberta_conv/evaluate.py

import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Paths / config ───────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("DATA_DIR", "./data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs/roberta_conv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model")
MAX_LENGTH = 128
BATCH_SIZE = 32
LABEL_NAMES = ["non-toxic", "toxic"]  # 0, 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Data ─────────────────────────────────────────────────────────────────────
test_df = pd.read_csv(TEST_PATH)
print(f"Test: {len(test_df)} rows | balance: {test_df['label'].value_counts().to_dict()}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()


class CommentDataset(Dataset):
    def __init__(self, df):
        self.encodings = tokenizer(
            df["text"].astype(str).tolist(),
            truncation=True, padding="max_length",
            max_length=MAX_LENGTH, return_tensors="pt",
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


test_loader = DataLoader(CommentDataset(test_df), batch_size=BATCH_SIZE)

# ── Inference ────────────────────────────────────────────────────────────────
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=-1)[:, 1]  # P(toxic)
        all_preds.extend(logits.argmax(-1).cpu().tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# ── Metrics ──────────────────────────────────────────────────────────────────
macro_f1 = f1_score(all_labels, all_preds, average="macro")
micro_f1 = f1_score(all_labels, all_preds, average="micro")
weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
try:
    auc = roc_auc_score(all_labels, all_probs)
except ValueError:
    auc = float("nan")

prec, rec, f1, support = precision_recall_fscore_support(
    all_labels, all_preds, labels=[0, 1], zero_division=0
)

print("\n" + "=" * 60)
print("FINAL METRICS ON TEST")
print("=" * 60)
print(f"Macro-F1:     {macro_f1:.4f}")
print(f"Micro-F1:     {micro_f1:.4f}")
print(f"Weighted-F1:  {weighted_f1:.4f}")
print(f"ROC-AUC:      {auc:.4f}")

print("\nPer class:")
print(f"{'class':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
for i, name in enumerate(LABEL_NAMES):
    print(f"{name:<12}{prec[i]:>10.4f}{rec[i]:>10.4f}{f1[i]:>10.4f}{support[i]:>10}")

print("\nClassification report (sklearn):")
print(classification_report(all_labels, all_preds, target_names=LABEL_NAMES, digits=4, zero_division=0))

# ── Confusion matrix ─────────────────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
print("Confusion matrix (rows = true, columns = predicted):")
print(f"{'':<14}{'pred non-tox':>14}{'pred toxic':>14}")
print(f"{'true non-tox':<14}{cm[0,0]:>14}{cm[0,1]:>14}")
print(f"{'true toxic':<14}{cm[1,0]:>14}{cm[1,1]:>14}")

print(f"\nFalse positives (non-toxic -> toxic): {cm[0,1]}")
print(f"Missed toxic (toxic -> non-toxic):    {cm[1,0]}")

# ── Save errors for manual inspection ────────────────────────────────────────
test_df = test_df.reset_index(drop=True)
test_df["pred"] = all_preds
test_df["prob_toxic"] = all_probs.round(4)
errors = test_df[test_df["label"] != test_df["pred"]].copy()
err_path = os.path.join(OUTPUT_DIR, "test_errors.csv")
errors.to_csv(err_path, index=False)
print(f"\nTotal errors: {len(errors)} of {len(test_df)}")
print(f"Saved for inspection: {err_path}")

# ── Optional W&B logging ─────────────────────────────────────────────────────
USE_WANDB = os.environ.get("USE_WANDB", "false").lower() == "true"
if USE_WANDB:
    import wandb

    wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),  # None if not set
        project="kazakh-toxicity-classification",
        name="evaluate-test",
        job_type="eval",
    )
    wandb.log({
        "test/macro_f1": macro_f1,
        "test/micro_f1": micro_f1,
        "test/weighted_f1": weighted_f1,
        "test/roc_auc": auc,
        "test/precision_nontoxic": prec[0], "test/recall_nontoxic": rec[0], "test/f1_nontoxic": f1[0],
        "test/precision_toxic": prec[1], "test/recall_toxic": rec[1], "test/f1_toxic": f1[1],
        "test/false_positives": int(cm[0, 1]),
        "test/false_negatives": int(cm[1, 0]),
    })
    wandb.log({"test/confusion_matrix": wandb.plot.confusion_matrix(
        y_true=all_labels.tolist(), preds=all_preds.tolist(), class_names=LABEL_NAMES
    )})
    wandb.finish()

print("\nDone.")
