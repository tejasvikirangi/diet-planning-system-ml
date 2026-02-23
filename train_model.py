import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib, os

# Sample training data (synthetic but valid)
data = {
    "bmi": [16, 18, 22, 25, 30, 35],
    "exercise": [0, 1, 2, 1, 0, 0],
    "goal": [
        "Weight Gain",
        "Weight Gain",
        "Maintain Weight",
        "Weight Loss",
        "Weight Loss",
        "Weight Loss"
    ]
}

df = pd.DataFrame(data)

X = df[["bmi", "exercise"]]
y = df["goal"]

model = DecisionTreeClassifier()
model.fit(X, y)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/diet_model.pkl")

print("✅ Diet prediction model trained successfully")
