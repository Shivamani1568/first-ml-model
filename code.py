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
model = KNeighborsClassifier(n_neighbors=1)
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