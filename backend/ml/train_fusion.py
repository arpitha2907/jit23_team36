import csv
import os
import re

import cv2
import joblib
import numpy as np
import pytesseract

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

DATA_DIR = os.path.join("data", "icdar", "training")
GT_PATH = os.path.join(DATA_DIR, "gt.txt")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fusion_model.joblib")

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# --------------------------------------------------
# TEXT NORMALIZATION
# --------------------------------------------------

def normalize_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


# --------------------------------------------------
# OCR
# --------------------------------------------------

def run_ocr(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return "", 0.0

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    data = pytesseract.image_to_data(
        rgb,
        config="--psm 7",
        output_type=pytesseract.Output.DICT,
    )

    words = []
    confidences = []

    for text, conf in zip(data["text"], data["conf"]):
        text = text.strip()

        try:
            conf = float(conf)
        except (ValueError, TypeError):
            continue

        if text and conf >= 0:
            words.append(text)
            confidences.append(conf)

    predicted = " ".join(words)

    if confidences:
        confidence = float(np.mean(confidences))
    else:
        confidence = 0.0

    return predicted, confidence


# --------------------------------------------------
# FEATURE GENERATION
# --------------------------------------------------

def create_features(ocr_text, confidence):
    """
    Generate four features compatible with fusion.py:

    1. normalized OCR confidence risk
    2. rule flag
    3. NLP flag
    4. normalized surprisal-like signal
    """

    confidence_risk = 1.0 - max(0.0, min(confidence, 100.0)) / 100.0

    # Simple suspicious-character / empty-output rule.
    rule_flag = float(
        not ocr_text.strip()
        or bool(re.search(r"[^\w\s\-?!.&'/]", ocr_text))
    )

    # Lightweight NLP-style signal.
    # Very short/empty OCR outputs are more suspicious.
    nlp_flag = float(len(ocr_text.strip()) <= 1)

    # Confidence-derived proxy for contextual uncertainty.
    # Kept in 0..1 for compatibility with the fusion feature vector.
    surprisal = confidence_risk

    return [
        confidence_risk,
        rule_flag,
        nlp_flag,
        surprisal,
    ]


# --------------------------------------------------
# LOAD ICDAR GROUND TRUTH
# --------------------------------------------------

def load_ground_truth():
    samples = []

    with open(GT_PATH, "r", encoding="utf-8-sig") as file:
        reader = csv.reader(file, skipinitialspace=True)

        for row in reader:
            if len(row) < 2:
                continue

            filename = row[0].strip()

            # Preserve commas if they occur inside transcription.
            ground_truth = ",".join(row[1:]).strip()

            samples.append((filename, ground_truth))

    return samples


# --------------------------------------------------
# BUILD TRAINING DATASET
# --------------------------------------------------

def build_dataset():
    samples = load_ground_truth()

    X = []
    y = []

    total = len(samples)

    print(f"Ground-truth entries: {total}")
    print("Running Tesseract OCR...")

    processed = 0
    missing = 0

    for index, (filename, ground_truth) in enumerate(samples, start=1):

        image_path = os.path.join(DATA_DIR, filename)

        if not os.path.exists(image_path):
            missing += 1
            continue

        ocr_text, confidence = run_ocr(image_path)

        correct = (
            normalize_text(ocr_text)
            == normalize_text(ground_truth)
        )

        # 0 = OCR correct
        # 1 = OCR error
        label = 0 if correct else 1

        features = create_features(
            ocr_text,
            confidence,
        )

        X.append(features)
        y.append(label)

        processed += 1

        if index % 250 == 0:
            print(
                f"Processed {index}/{total} "
                f"(usable={processed}, missing={missing})"
            )

    return np.array(X), np.array(y)


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

def train():
    X, y = build_dataset()

    print()
    print("Dataset created")
    print("-------------------------")
    print("Samples:", len(X))
    print("Correct OCR:", int(np.sum(y == 0)))
    print("OCR errors:", int(np.sum(y == 1)))

    if len(np.unique(y)) < 2:
        raise RuntimeError(
            "Training requires both correct and incorrect OCR samples."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print()
    print("Training Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print()
    print("Evaluation")
    print("=========================")

    print(f"Accuracy: {accuracy:.4f}")

    print()
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print()
    print("Classification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
        )
    )

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print()
    print("=========================")
    print("Model saved successfully")
    print(MODEL_PATH)


if __name__ == "__main__":
    train()