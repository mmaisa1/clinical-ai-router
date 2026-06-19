# train.py
# Trains TF-IDF + LogReg and TF-IDF + LinearSVC (calibrated) on TCGA
# pathology reports and serializes the models to disk.
#
# Usage:
#   python train.py
#   python train.py --data_dir ./data --models_dir ./models
#
# Input:
#   data/TCGA_Reports.csv
#   data/tcga_patient_to_cancer_type.csv
#
# Output:
#   models/vectorizer.joblib
#   models/svm_model.joblib
#   models/lr_model.joblib
#
# These files are loaded by the API at startup.
# Run this once before starting the API for the first time,
# or whenever you retrain on new data.

import argparse
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import LinearSVC

# Reproducibility 
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def main(data_dir: str, models_dir: str) -> None:
    data_dir   = Path(data_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    reports_path = data_dir / "TCGA_Reports.csv"
    meta_path    = data_dir / "tcga_patient_to_cancer_type.csv"

    if not reports_path.exists():
        raise FileNotFoundError(
            f"Missing: {reports_path}\n"
            "Download TCGA data and place in the data/ directory."
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing: {meta_path}\n"
            "Download TCGA metadata and place in the data/ directory."
        )

    reports_df = pd.read_csv(reports_path)
    reports_df["patient_id"] = (
        reports_df["patient_filename"].astype(str).str.split(".").str[0]
    )
    meta_df = pd.read_csv(meta_path)

    tcga_df = (
        reports_df
        .merge(meta_df, on="patient_id", how="inner")
        .dropna(subset=["text", "cancer_type"])
        .reset_index(drop=True)
    )

    print(f"Loaded {len(tcga_df):,} reports | "
          f"{tcga_df['cancer_type'].nunique()} cancer types | "
          f"{tcga_df['patient_id'].nunique():,} unique patients")

    max_per_patient = tcga_df.groupby("patient_id").size().max()
    print(f"Max reports per patient: {max_per_patient}")

    # Patient-disjoint train / test split
    # SEED=42 is kept fixed to match the offline evaluation setup.
    X      = tcga_df["text"]
    y      = tcga_df["cancer_type"]
    groups = tcga_df["patient_id"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    overlap = len(
        set(tcga_df["patient_id"].iloc[train_idx]) &
        set(tcga_df["patient_id"].iloc[test_idx])
    )
    assert overlap == 0, "Patient leakage detected in split."
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,} | "
          f"Patient overlap: 0 ✓")

    # TF-IDF vectorizer
    # Trains on full report text — consistent with how the API uses the
    # vectorizer at inference time (Stage 1 candidate narrowing).
    # Negation terms kept — removing "no", "not" etc. loses clinical signal.
    NEGATION_KEEP = {"no", "not", "nor", "never", "without"}
    stopwords     = sorted(set(ENGLISH_STOP_WORDS) - NEGATION_KEEP)

    vectorizer = TfidfVectorizer(
        stop_words=stopwords,
        max_features=50_000,
        ngram_range=(1, 2),
        min_df=2,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)
    print(f"TF-IDF vocabulary: {len(vectorizer.vocabulary_):,} features")

    # Train models
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(
        solver="saga",
        max_iter=6000,
        class_weight="balanced",
        random_state=SEED,
        C=4.0,
    )
    lr_model.fit(X_train_tfidf, y_train)

    # LinearSVC wrapped in CalibratedClassifierCV to produce predict_proba().
    # The API needs probability-like scores for routing decisions.
    # These scores are used as routing signals, not calibrated probabilities.
    print("Training LinearSVC (calibrated)...")
    svm_base  = LinearSVC(random_state=SEED, class_weight="balanced")
    svm_model = CalibratedClassifierCV(svm_base, cv=5)
    svm_model.fit(X_train_tfidf, y_train)

    # Evaluate on held-out test set
    print("\nTest set results")
    for name, model in [("LogReg", lr_model), ("LinearSVC (calibrated)", svm_model)]:
        preds = model.predict(X_test_tfidf)
        acc   = accuracy_score(y_test, preds)
        mf1   = f1_score(y_test, preds, average="macro", zero_division=0)
        wf1   = f1_score(y_test, preds, average="weighted", zero_division=0)
        print(f"{name:<25}  acc={acc:.4f}  "
              f"macro-F1={mf1:.4f}  weighted-F1={wf1:.4f}")

    # Serialize
    joblib.dump(vectorizer, models_dir / "vectorizer.joblib")
    joblib.dump(lr_model,   models_dir / "lr_model.joblib")
    joblib.dump(svm_model,  models_dir / "svm_model.joblib")

    print(f"\nSaved to {models_dir}/")
    print("  vectorizer.joblib ✓")
    print("  lr_model.joblib   ✓")
    print("  svm_model.joblib  ✓")

    # Verify
    print("\nVerifying saved models load correctly...")
    vec_check = joblib.load(models_dir / "vectorizer.joblib")
    svm_check = joblib.load(models_dir / "svm_model.joblib")

    sample       = ["breast ductal invasive carcinoma mastectomy sentinel node"]
    sample_tfidf = vec_check.transform(sample)
    pred         = svm_check.predict(sample_tfidf)[0]
    score        = svm_check.predict_proba(sample_tfidf).max()

    print(f"  Sample prediction : {pred}")
    print(f"  Routing score     : {score:.4f}")
    print("\ntrain.py complete. Models are ready for the API.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and serialize models for the Clinical AI Inference Router."
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Directory containing TCGA_Reports.csv and tcga_patient_to_cancer_type.csv"
    )
    parser.add_argument(
        "--models_dir",
        default="./models",
        help="Directory where serialized model files will be saved"
    )
    args = parser.parse_args()
    main(args.data_dir, args.models_dir)