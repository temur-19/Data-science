import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

from pathlib import Path
DATA_DIR = Path(__file__).parent / "data"

csv_files = sorted(DATA_DIR.glob("*.csv"))

print("Topilgan CSV fayllar:")

for file in csv_files:
    print(file.name)

from sklearn.pipeline import make_pipeline


fayl = Path(__file__).parent / "data" / "diabetes_binary_health_indicators_BRFSS2015.csv"
df = pd.read_csv(fayl)

X = df[["HighBP", "HighChol", "BMI", "Age", "PhysActivity"]]
y = df["Diabetes_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = make_pipeline(
    SimpleImputer(strategy="median"),
    LogisticRegression(max_iter=2000)
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, pred), 2))

ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.title("Diabet: Logistic Regression")
plt.show()
plt.savefig('diabet/data/result.png')

test = pd.DataFrame({
    "HighBP":       [0, 1, 1, 0, 1, 1],
    "HighChol":     [0, 1, 1, 0, 1, 1],
    "BMI":          [22, 32, 35, 25, 30, 50],
    "Age":          [2, 9, 11, 5, 8, 60],
    "PhysActivity": [1, 0, 0, 1, 1, 1]
})

bashorat = model.predict(test)
ehtimol = model.predict_proba(test)[:, 1]

for i in range(6):
    print(f"\n{i + 1}-sun'iy namuna")
    if round(ehtimol[i], 2)>=0.5:
        print("Diabet bashorati (0/1):", bashorat[i])
        print("Diabet ehtimoli:", round(ehtimol[i], 2))
        print('Bemorda diabet ehtimoli bor')

    elif round(ehtimol[i], 2)<0.5:
        print("Diabet bashorati (0/1):", bashorat[i])
        print("Diabet ehtimoli:", round(ehtimol[i], 2))
        print("Bemorda diabet ehtimoli yo'q")
