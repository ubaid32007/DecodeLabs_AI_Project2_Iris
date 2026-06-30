"""
predict.py - Load the trained KNN model and classify new Iris samples.
Run: python3 predict.py
"""
import joblib
import os
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "knn_iris_model.pkl")


def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["scaler"], bundle["target_names"], bundle["feature_names"]


def predict_sample(sepal_length, sepal_width, petal_length, petal_width):
    model, scaler, target_names, feature_names = load_model()
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    sample_scaled = scaler.transform(sample)
    pred_idx = model.predict(sample_scaled)[0]
    probs = model.predict_proba(sample_scaled)[0] if hasattr(model, "predict_proba") else None
    return target_names[pred_idx], probs, target_names


if __name__ == "__main__":
    print("Enter flower measurements (cm):")
    sl = float(input("Sepal length: "))
    sw = float(input("Sepal width: "))
    pl = float(input("Petal length: "))
    pw = float(input("Petal width: "))

    species, probs, target_names = predict_sample(sl, sw, pl, pw)
    print(f"\nPredicted species: {species}")
    if probs is not None:
        for name, p in zip(target_names, probs):
            print(f"  {name}: {p*100:.1f}%")
