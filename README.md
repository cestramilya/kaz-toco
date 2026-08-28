# Kazakh Toxic Comment Classification

Binary toxicity classification (toxic / non-toxic) of Kazakh-language YouTube comments. The repository contains two models:

- **TF-IDF + Logistic Regression** — an interpretable baseline combining word-level (1–2 grams) and character-level (2–5 grams) TF-IDF features, with the regularization strength `C` selected by validation macro-F1.
- **KazRoBERTa (conversational)** — fine-tuning of the public [`kz-transformers/kaz-roberta-conversational`](https://huggingface.co/kz-transformers/kaz-roberta-conversational) checkpoint with a two-stage schedule: classifier-head warmup with a frozen encoder, then full fine-tuning with early stopping on validation macro-F1.

## Repository structure

```
repo/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── train.csv.gz
│   ├── val.csv
│   └── test.csv
└── models/
    ├── tfidf_logreg/
    │   ├── train.py
    │   └── evaluate.py
    └── roberta_conv/
        ├── train.py
        └── evaluate.py
```

## Setup

Python 3.10+ is recommended.

```
pip install -r requirements.txt
```

`wandb` is an **optional** dependency (see "Weights & Biases logging" below); everything runs without it, but it is included in `requirements.txt` for convenience.

## Data

Place the dataset splits in `data/` as `train.csv.gz`, `val.csv`, and `test.csv` (already included in this repository). The training split is gzip-compressed to stay under hosting file-size limits; `pandas.read_csv` reads it transparently, no manual decompression needed. Each CSV has the columns:

`video_id, text, like_count, is_reply, is_pinned, is_favorited, label`

where `label` is `0` (non-toxic) or `1` (toxic). The test split contains comments from 5 videos that are entirely held out from training and validation.

By default the scripts read data from `./data` relative to the repository root. To use a different location, set the `DATA_DIR` environment variable.

## Running the models

All commands are run **from the repository root**. Trained artifacts are written to `./outputs/<model>/` (override with the `OUTPUT_DIR` environment variable).

### TF-IDF + Logistic Regression baseline

```
python models/tfidf_logreg/train.py
python models/tfidf_logreg/evaluate.py
```

`train.py` fits the vectorizers, selects `C` on the validation split, and saves `word_vectorizer.joblib`, `char_vectorizer.joblib`, and `logreg_model.joblib` to `./outputs/tfidf_logreg/`. `evaluate.py` loads these artifacts and reports the final test metrics (macro/micro/weighted F1, ROC-AUC, per-class precision/recall, confusion matrix), and saves misclassified examples to `test_errors_baseline.csv`.

### KazRoBERTa fine-tuning

```
python models/roberta_conv/train.py
python models/roberta_conv/evaluate.py
```

`train.py` downloads the pretrained checkpoint from the Hugging Face Hub, fine-tunes it, and saves the best checkpoint (by validation macro-F1) to `./outputs/roberta_conv/best_model/`. `evaluate.py` loads that checkpoint and reports the final test metrics. A CUDA GPU is strongly recommended for training (mixed precision is enabled automatically on GPU); evaluation also works on CPU, just slower.

## Weights & Biases logging (optional)

W&B logging is **disabled by default** — no account, login, or network access to W&B is required to run any script. To enable it:

```
USE_WANDB=true WANDB_ENTITY=<your-entity> python models/roberta_conv/train.py
```

(On Windows PowerShell: `$env:USE_WANDB="true"; $env:WANDB_ENTITY="<your-entity>"; python models\roberta_conv\train.py`.)

If `WANDB_ENTITY` is not set, W&B uses the default entity of the logged-in account.

## Reproducibility

All scripts fix the random seed (`SEED = 42`). Note that GPU nondeterminism in cuDNN kernels may still cause small run-to-run variation in transformer fine-tuning.
