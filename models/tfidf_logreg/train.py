# Train the TF-IDF + logistic regression baseline for Kazakh toxic comment
# classification. Run from the repository root:
#   python models/tfidf_logreg/train.py
# Optional W&B logging:
#   USE_WANDB=true WANDB_ENTITY=<your-entity> python models/tfidf_logreg/train.py

import os
import random

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("DATA_DIR", "./data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs/tfidf_logreg")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv.gz")  # pandas reads gzip transparently
VAL_PATH = os.path.join(DATA_DIR, "val.csv")

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Config ───────────────────────────────────────────────────────────────────
CFG = {
    # word-level TF-IDF
    "word_ngram_range": (1, 2),
    "word_max_features": 100_000,
    "word_min_df": 2,
    # character-level TF-IDF (important for Kazakh: rich morphology)
    "char_ngram_range": (2, 5),
    "char_max_features": 200_000,
    "char_min_df": 2,
    # logistic regression
    "C_grid": [0.1, 0.5, 1.0, 3.0, 10.0],  # selected by val macro-F1
    "max_iter": 2000,
    "class_weight": "balanced",
    "solver": "liblinear",
    "seed": SEED,
}

# ── Optional W&B logging ─────────────────────────────────────────────────────
USE_WANDB = os.environ.get("USE_WANDB", "false").lower() == "true"
if USE_WANDB:
    import wandb

    wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),  # None if not set
        project="kazakh-toxicity-classification",
        name="baseline-tfidf-logreg-v1",
        config=CFG,
    )

# ── Data ─────────────────────────────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

X_train_text = train_df["text"].astype(str).tolist()
y_train = train_df["label"].values
X_val_text = val_df["text"].astype(str).tolist()
y_val = val_df["label"].values

# class balance check
print("Train balance:", train_df["label"].value_counts().to_dict())
print("Val balance:  ", val_df["label"].value_counts().to_dict())

# ── Vectorization: word + char TF-IDF ────────────────────────────────────────
print("\n=== Vectorization ===")
word_vec = TfidfVectorizer(
    analyzer="word",
    ngram_range=CFG["word_ngram_range"],
    max_features=CFG["word_max_features"],
    min_df=CFG["word_min_df"],
    sublinear_tf=True,
    lowercase=True,
)
char_vec = TfidfVectorizer(
    analyzer="char_wb",  # character n-grams within word boundaries
    ngram_range=CFG["char_ngram_range"],
    max_features=CFG["char_max_features"],
    min_df=CFG["char_min_df"],
    sublinear_tf=True,
    lowercase=True,
)

X_train_word = word_vec.fit_transform(X_train_text)
X_train_char = char_vec.fit_transform(X_train_text)
X_val_word = word_vec.transform(X_val_text)
X_val_char = char_vec.transform(X_val_text)

X_train = hstack([X_train_word, X_train_char]).tocsr()
X_val = hstack([X_val_word, X_val_char]).tocsr()

print(f"Word vocabulary size: {len(word_vec.vocabulary_)}")
print(f"Char vocabulary size: {len(char_vec.vocabulary_)}")
print(f"Train matrix: {X_train.shape} | val: {X_val.shape}")
if USE_WANDB:
    wandb.log({
        "features/word_vocab_size": len(word_vec.vocabulary_),
        "features/char_vocab_size": len(char_vec.vocabulary_),
        "features/total_dim": X_train.shape[1],
    })

# ── Grid search over C by val macro-F1 ───────────────────────────────────────
print("\n=== Logistic regression C search ===")
best_val_f1, best_C, best_model = -1.0, None, None

for C in CFG["C_grid"]:
    clf = LogisticRegression(
        C=C,
        max_iter=CFG["max_iter"],
        class_weight=CFG["class_weight"],
        solver=CFG["solver"],
        random_state=SEED,
    )
    clf.fit(X_train, y_train)

    val_preds = clf.predict(X_val)
    val_f1 = f1_score(y_val, val_preds, average="macro")
    val_acc = accuracy_score(y_val, val_preds)

    if USE_WANDB:
        wandb.log({"grid/C": C, "grid/val_f1": val_f1, "grid/val_acc": val_acc})
    print(f"C={C:<5} — val_f1: {val_f1:.4f} | val_acc: {val_acc:.4f}")

    if val_f1 > best_val_f1:  # maximize F1, matching the transformer pipeline
        best_val_f1, best_C, best_model = val_f1, C, clf
        print(f"  -> new best model (C={C}, val_f1={val_f1:.4f})")

# ── Final metrics of the best model on val ───────────────────────────────────
print(f"\n=== Best model: C={best_C} ===")
val_preds = best_model.predict(X_val)

val_metrics = {
    "best/C": best_C,
    "best/val_f1_macro": f1_score(y_val, val_preds, average="macro"),
    "best/val_f1_weighted": f1_score(y_val, val_preds, average="weighted"),
    "best/val_accuracy": accuracy_score(y_val, val_preds),
    "best/val_precision_macro": precision_score(y_val, val_preds, average="macro"),
    "best/val_recall_macro": recall_score(y_val, val_preds, average="macro"),
}
if USE_WANDB:
    wandb.log(val_metrics)

print(classification_report(y_val, val_preds, digits=4))
print("Confusion matrix:")
print(confusion_matrix(y_val, val_preds))

if USE_WANDB:
    wandb.log({"best/confusion_matrix": wandb.plot.confusion_matrix(
        y_true=y_val.tolist(), preds=val_preds.tolist(),
        class_names=[str(c) for c in sorted(set(y_val))],
    )})

# ── Top features by logreg weights (baseline interpretability) ───────────────
feature_names = np.array(
    word_vec.get_feature_names_out().tolist()
    + char_vec.get_feature_names_out().tolist()
)
coefs = best_model.coef_[0]
top_pos = feature_names[np.argsort(coefs)[-20:]][::-1]
top_neg = feature_names[np.argsort(coefs)[:20]]
print("\nTop-20 features for class 1 (toxic):    ", top_pos.tolist())
print("Top-20 features for class 0 (non-toxic):", top_neg.tolist())
if USE_WANDB:
    wandb.log({
        "interpret/top_features_class1": wandb.Table(
            columns=["feature"], data=[[f] for f in top_pos]),
        "interpret/top_features_class0": wandb.Table(
            columns=["feature"], data=[[f] for f in top_neg]),
    })

# ── Save the baseline ────────────────────────────────────────────────────────
joblib.dump(word_vec, os.path.join(OUTPUT_DIR, "word_vectorizer.joblib"))
joblib.dump(char_vec, os.path.join(OUTPUT_DIR, "char_vectorizer.joblib"))
joblib.dump(best_model, os.path.join(OUTPUT_DIR, "logreg_model.joblib"))
print(f"\nBaseline saved to: {OUTPUT_DIR}")

if USE_WANDB:
    wandb.log({"final/best_val_f1": best_val_f1})
    wandb.finish()
print(f"Done. Best baseline val_f1: {best_val_f1:.4f}")
print("Compare with the transformer's val_f1 to estimate the gain from the transformer.")
