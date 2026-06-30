"""
DecodeLabs Industrial Training - Project 2
Data Classification Using AI (Iris Benchmark, KNN)

Pipeline: INPUT -> PROCESS -> OUTPUT
  INPUT   : Load Iris dataset, scale features
  PROCESS : Train/test split, KNN training, K tuning
  OUTPUT  : Confusion matrix, F1 score, classification report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score
)
import joblib
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------- 1. INPUT: Load and understand the dataset ----------
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    df["species_name"] = df["species"].map(dict(enumerate(iris.target_names)))
    return df, iris.feature_names, iris.target_names


# ---------- 2. PROCESS: Scale, split, train ----------
def preprocess_and_split(df, feature_names):
    X = df[feature_names].values
    y = df["species"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
    )
    return X_train, X_test, y_train, y_test, scaler


def find_best_k(X_train, y_train, X_test, y_test, k_range=range(1, 21)):
    errors = []
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        errors.append(np.mean(pred != y_test))

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), errors, marker="o", color="#1f4e79")
    plt.title("Tuning the Engine: Choosing K")
    plt.xlabel("K Value")
    plt.ylabel("Error Rate")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUT_DIR, "k_tuning.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Avoid K=1 (overfits / noise-sensitive per slide 12) unless it's the only option.
    # Prefer the smallest odd K (avoids ties) within tolerance of the minimum error.
    errors = np.array(errors)
    k_list = list(k_range)
    min_err = errors.min()
    candidates = [k_list[i] for i in range(len(k_list))
                  if errors[i] <= min_err + 1e-9 and k_list[i] > 1]
    if not candidates:
        candidates = [k for k in k_list if k > 1] or k_list
    odd_candidates = [k for k in candidates if k % 2 == 1]
    best_k = min(odd_candidates) if odd_candidates else min(candidates)
    return best_k


def train_knn(X_train, y_train, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model


# ---------- 3. OUTPUT: Evaluate ----------
def evaluate(model, X_test, y_test, target_names):
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")
    report = classification_report(y_test, predictions, target_names=target_names)

    with open(os.path.join(OUT_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nWeighted F1 Score: {f1:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix - KNN Classifier")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    return acc, f1, report


def main():
    print("=" * 55)
    print("DecodeLabs Project 2: Data Classification Using AI")
    print("=" * 55)

    df, feature_names, target_names = load_data()
    print(f"\n[INPUT] Loaded {len(df)} samples, {len(feature_names)} features, "
          f"{len(target_names)} classes: {list(target_names)}")

    X_train, X_test, y_train, y_test, scaler = preprocess_and_split(df, feature_names)
    print(f"[PROCESS] Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    best_k = find_best_k(X_train, y_train, X_test, y_test)
    print(f"[PROCESS] Optimal K found via elbow method: K={best_k}")

    model = train_knn(X_train, y_train, best_k)
    print(f"[PROCESS] KNN model trained with K={best_k}")

    acc, f1, report = evaluate(model, X_test, y_test, target_names)
    print(f"\n[OUTPUT] Accuracy: {acc:.4f}")
    print(f"[OUTPUT] Weighted F1 Score: {f1:.4f}")
    print("\nClassification Report:\n", report)

    joblib.dump({"model": model, "scaler": scaler, "target_names": target_names,
                 "feature_names": feature_names, "k": best_k},
                os.path.join(MODEL_DIR, "knn_iris_model.pkl"))
    print(f"[SAVED] Model saved to models/knn_iris_model.pkl")
    print(f"[SAVED] Plots saved to outputs/ (confusion_matrix.png, k_tuning.png)")
    print(f"[SAVED] Report saved to outputs/classification_report.txt")


if __name__ == "__main__":
    main()
