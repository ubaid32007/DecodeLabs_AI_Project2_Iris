# Project 2: Data Classification Using AI
**DecodeLabs Industrial Training Kit — Batch 2026**

## Goal
Build a classification model that learns to distinguish between three species
of Iris flowers (Setosa, Versicolor, Virginica) from four numeric measurements,
using the K-Nearest Neighbors (KNN) supervised learning algorithm.

## Pipeline (IPO Framework)

| Stage   | What happens |
|---------|--------------|
| INPUT   | Load the 150-sample Iris dataset (4 features, 3 balanced classes), scale features with `StandardScaler` |
| PROCESS | Shuffle + split into 80% train / 20% test, tune K via elbow method, train `KNeighborsClassifier` |
| OUTPUT  | Confusion matrix, classification report (precision/recall/F1), accuracy |

## Project Structure
```
AI_Project2_Iris/
├── src/
│   ├── main.py       # full training + evaluation pipeline
│   └── predict.py     # CLI tool to classify a new flower sample
├── outputs/
│   ├── confusion_matrix.png
│   ├── k_tuning.png
│   └── classification_report.txt
├── models/
│   └── knn_iris_model.pkl
└── README.md
```

## How to Run
```bash
pip install scikit-learn pandas matplotlib seaborn joblib

# Train and evaluate
python3 src/main.py

# Predict a new sample interactively
python3 src/predict.py
```

## Results
- **Algorithm:** K-Nearest Neighbors
- **Best K:** chosen automatically via elbow-method error-rate scan, avoiding K=1 (overfitting risk)
- **Accuracy:** ~96.7% on held-out test set
- **Weighted F1 Score:** ~0.97

See `outputs/classification_report.txt` and `outputs/confusion_matrix.png` for full results.

## Key Concepts Demonstrated
- Feature scaling (StandardScaler) — required for distance-based algorithms like KNN
- Train/test split with stratification and shuffling to remove order bias
- Hyperparameter tuning (choosing K)
- Confusion matrix as a diagnostic tool (TP/FP/FN/TN)
- Precision/Recall trade-off and F1 score as the harmonic mean
- Why raw accuracy can be misleading on imbalanced data (not an issue here since Iris is balanced, but documented per the training material)

## Next Steps (per training material)
This tabular classification pipeline is the foundation for later modules
covering deep learning and computer vision (CNNs on image data).
