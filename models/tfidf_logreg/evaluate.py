# Final evaluation of the TF-IDF + logistic regression baseline on the
# held-out test set (comments from 5 videos never seen during training).
# Run from the repository root, after models/tfidf_logreg/train.py:
#   python models/tfidf_logreg/evaluate.py

import os

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# ── Paths / config ───────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("DATA_DIR", "./data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs/tfidf_logreg")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
LABEL_NAMES = ["non-toxic", "toxic"]  # 0, 1

# ── Data ─────────────────────────────────────────────────────────────────────
test_df = pd.read_csv(TEST_PATH)
print(f"Test: {len(test_df)} rows | balance: {test_df['label'].value_counts().to_dict()}")

X_test_text = test_df["text"].astype(str).tolist()
all_labels = test_df["label"].values

# ── Load the trained baseline ────────────────────────────────────────────────
word_vec = joblib.load(os.path.join(OUTPUT_DIR, "word_vectorizer.joblib"))
char_vec = joblib.load(os.path.join(OUTPUT_DIR, "char_vectorizer.joblib"))
model = joblib.load(os.path.join(OUTPUT_DIR, "logreg_model.joblib"))
print("Baseline loaded (word_vec + char_vec + logreg).")

# ── Inference ────────────────────────────────────────────────────────────────
X_test_word = word_vec.transform(X_test_text)
X_test_char = char_vec.transform(X_test_text)
X_test = hstack([X_test_word, X_test_char]).tocsr()
print(f"Test matrix: {X_test.shape}")

all_preds = model.predict(X_test)
all_probs = model.predict_proba(X_test)[:, 1]  # P(toxic)

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
print("FINAL BASELINE METRICS ON TEST")
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
err_path = os.path.join(OUTPUT_DIR, "test_errors_baseline.csv")
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
        name="evaluate-test-baseline",
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

print("\nDone. Compare test/macro_f1 with the transformer's evaluate-test run.")
