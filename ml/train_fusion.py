"""Trains the signal-fusion classifier used by
backend/app/services/fusion.py.

Expects a CSV with columns: ocr_conf, rule_flag, nlp_flag, ml_surprisal, label
(label = 1 if the token is an OCR error, 0 otherwise). Build that CSV by
running the ICDAR 2017/2019 post-OCR correction dataset's aligned OCR /
ground-truth pairs through the same three detection layers used in
production (backend/app/services/{rules,nlp_check,ml_context}.py), then
labeling each token as correct/incorrect from the alignment.

Usage:
    python ml/train_fusion.py --data path/to/labeled_features.csv \
        --out ../backend/models/fusion_model.joblib

If --data is omitted, a synthetic dataset is generated instead, purely so
this script (and the rest of the pipeline) can be smoke-tested end-to-end
before the real ICDAR-derived training set is ready.
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def synthetic_dataset(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ocr_conf = rng.uniform(0, 100, n)
    rule_flag = rng.integers(0, 2, n)
    nlp_flag = rng.integers(0, 2, n)
    ml_surprisal = rng.exponential(2.0, n)

    norm_conf = 1 - ocr_conf / 100
    norm_surprisal = 1 - np.exp(-ml_surprisal)
    risk = 0.35 * norm_conf + 0.25 * rule_flag + 0.15 * nlp_flag + 0.25 * norm_surprisal
    label = (risk + rng.normal(0, 0.1, n) > 0.5).astype(int)

    return pd.DataFrame(
        {
            "ocr_conf": ocr_conf,
            "rule_flag": rule_flag,
            "nlp_flag": nlp_flag,
            "ml_surprisal": ml_surprisal,
            "label": label,
        }
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    norm_conf = 1 - df["ocr_conf"].clip(0, 100) / 100
    norm_surprisal = 1 - np.exp(-df["ml_surprisal"])
    return pd.DataFrame(
        {
            "norm_conf": norm_conf,
            "rule_flag": df["rule_flag"].astype(float),
            "nlp_flag": df["nlp_flag"].astype(float),
            "norm_surprisal": norm_surprisal,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default=None, help="CSV with ocr_conf,rule_flag,nlp_flag,ml_surprisal,label"
    )
    parser.add_argument("--out", default="../backend/models/fusion_model.joblib")
    args = parser.parse_args()

    if args.data:
        df = pd.read_csv(args.data)
    else:
        print("No --data given; training on a synthetic dataset for smoke-testing only.")
        df = synthetic_dataset()

    X = build_features(df)
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(clf, args.out)
    print(f"Saved fusion model to {args.out}")


if __name__ == "__main__":
    main()
