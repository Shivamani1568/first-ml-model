# First Model — Iris Classification

My first end-to-end machine learning project. Trains a tiny model on the Iris flower dataset and completes the full ML loop:

```
Load data → Split data → Train model → Predict → Check accuracy
```

---

## What this project does

It uses the classic **Iris dataset** (built into scikit-learn) — 150 flowers with 4 measurements each, belonging to one of 3 species. The script:

1. Loads the dataset
2. Splits it into training (80%) and test (20%) data
3. Trains a **K-Nearest Neighbors** classifier (k=3)
4. Predicts species on the test flowers
5. Reports the accuracy
6. Tries one hand-crafted flower for fun

---

## Setup

### Requirements
- macOS (this project was built on a MacBook Air)
- Python 3.12 at `/usr/local/bin/python3`
- VS Code

### Install dependencies

In the VS Code terminal (`` Ctrl + ` `` to open):

```bash
/usr/local/bin/python3 -m pip install scikit-learn --break-system-packages
```

This installs `scikit-learn` along with its dependencies (`numpy`, `scipy`, `joblib`, `threadpoolctl`).

> The `--break-system-packages` flag is needed on macOS with Homebrew Python to bypass the "externally-managed-environment" restriction.

---

## How to run it

### Option 1 — Terminal command

```bash
/usr/local/bin/python3 /Users/karthiksanjeevuni/Desktop/project_2_files/2_py/code.py
```

### Option 2 — VS Code Run button

1. Open `code.py` in VS Code
2. Click the **▶ play button** at the top-right of the editor

---

## The code

`code.py`:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load data
iris = load_iris()
X = iris.data
y = iris.target

print("Feature names :", iris.feature_names)
print("Target names  :", list(iris.target_names))
print("X shape       :", X.shape)
print("y shape       :", y.shape)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training set:", X_train.shape[0], "samples")
print("Test set    :", X_test.shape[0], "samples")

# 3. Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)
print("Predicted:", y_pred)
print("Actual   :", y_test)

# 5. Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}  ({accuracy * 100:.2f}%)")

# 6. Try a single flower
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)
print("Predicted class :", prediction[0])
print("Predicted name  :", iris.target_names[prediction[0]])
```

---

## Output

```
Feature names : ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
Target names  : ['setosa', 'versicolor', 'virginica']
X shape       : (150, 4)
y shape       : (150,)
Training set: 120 samples
Test set    : 30 samples
Predicted: [0 2 1 1 0 1 0 0 2 1 2 2 2 1 0 0 0 1 1 2 0 2 1 2 2 1 1 0 2 0]
Actual   : [0 2 1 1 0 1 0 0 2 1 2 2 2 1 0 0 0 1 1 2 0 2 1 2 2 1 1 0 2 0]
Accuracy: 1.0000  (100.00%)
Predicted class : 0
Predicted name  : setosa
```

---

## What's happening, step by step

### 1. Load data
The Iris dataset has 150 flowers. Each flower has 4 measurements (sepal length, sepal width, petal length, petal width) and a species label (0 = setosa, 1 = versicolor, 2 = virginica).
- `X` — features (the inputs), shape `(150, 4)`
- `y` — target (the answer), shape `(150,)`

### 2. Split data
We split into two pieces:
- **Training set** (120 flowers) — the model learns from these
- **Test set** (30 flowers) — kept hidden, used only to grade the model later

`random_state=42` makes the split reproducible. `stratify=y` ensures both sets have a balanced mix of all 3 species.

### 3. Train model
**K-Nearest Neighbors (KNN)** with k=3: to classify a new flower, it finds the 3 closest flowers in the training set (by measurement distance) and takes a majority vote.

### 4. Predict
The trained model predicts species for the 30 test flowers it has never seen.

### 5. Check accuracy
Compares predictions to the real labels. On this seed, the model got **30 out of 30** correct → **100%**.

> Iris is famously easy — 100% is normal here. On real-world problems, expect 70–95%.

### 6. Single prediction
Hand-crafted flower `[5.1, 3.5, 1.4, 0.2]` → predicted **setosa**.

---

## Concepts learned

| Concept | Plain-English meaning |
|---|---|
| **Feature** | One measurable input (e.g. petal length). Columns of `X`. |
| **Target** | The answer to predict (the species). The vector `y`. |
| **Training data** | The 120 flowers the model learns from (`X_train`, `y_train`). |
| **Test data** | The 30 flowers held back to grade the model (`X_test`, `y_test`). |
| **Prediction** | The model's guess for each test flower. Compared to `y_test` to compute accuracy. |

---

## File structure

```
project_2_files/
└── 2_py/
    └── code.py        ← the script
```

---

## Notes for next time

- Run the script again only when the code changes — same seed gives the same output.
- Try changing `n_neighbors=3` to `n_neighbors=1` or `n_neighbors=50` and see how accuracy shifts. That's how you build intuition.
- The same 5-step loop (load → split → train → predict → check) works for almost every supervised ML problem — only the data and the model class change.
