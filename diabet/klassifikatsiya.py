import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print("X o'lchami:", X.shape)
print("Sinflar:", data.target_names)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

from sklearn.linear_model import LogisticRegression

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

log_model.fit(X_train_scaled, y_train)

log_pred = log_model.predict(X_test_scaled)

print("Accuracy :", round(accuracy_score(y_test, log_pred), 4))
print("Precision:", round(precision_score(y_test, log_pred), 4))
print("Recall   :", round(recall_score(y_test, log_pred), 4))
print("F1-score :", round(f1_score(y_test, log_pred), 4))

print(classification_report(
    y_test, log_pred,
    target_names=["Malignant", "Benign"]
))


fig, ax = plt.subplots(figsize=(6, 5))

ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, log_pred),
    display_labels=["Malignant (0)", "Benign (1)"]
).plot(ax=ax, values_format="d")

ax.set_title("Logistic Regression — Confusion Matrix")
plt.tight_layout()
plt.show()