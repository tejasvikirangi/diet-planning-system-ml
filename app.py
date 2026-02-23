
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model & data
model = joblib.load("model/diet_model.pkl")
food_df = pd.read_csv("data/food_dataset.csv")
food_df.columns = food_df.columns.str.lower()

# Disease rules
disease_rules = {
    "None": {
        "allowed": ["fruits", "vegetables", "whole grains"],
        "avoid": []
    },

    "Diabetes": {
        "allowed": ["vegetables", "nuts", "whole grains"],
        "avoid": ["sweets", "cakes", "sugary drinks"]
    },

    "Hypertension": {
        "allowed": ["fruits", "leafy vegetables"],
        "avoid": ["salt", "processed food"]
    },

    "Obesity": {
        "allowed": ["salads", "fruits", "vegetables"],
        "avoid": ["fast food", "fried food"]
    },

    "Heart Disease": {
        "allowed": ["fish", "oats", "fruits"],
        "avoid": ["butter", "red meat"]
    },

    "Anemia": {
        "allowed": ["spinach", "iron rich foods"],
        "avoid": ["junk food"]
    },

    "Kidney Disease": {
        "allowed": ["low potassium foods", "vegetables"],
        "avoid": ["salt", "high protein food"]
    },

    "Thyroid": {
        "allowed": ["iodine rich foods", "vegetables"],
        "avoid": ["soy", "processed food"]
    },

    "Gastritis": {
        "allowed": ["boiled vegetables", "fruits"],
        "avoid": ["spicy food", "fried food"]
    },

    "PCOS": {
        "allowed": ["fiber rich foods", "vegetables"],
        "avoid": ["sugar", "refined carbs"]
    },

    "Fatty Liver": {
        "allowed": ["fruits", "vegetables", "whole grains"],
        "avoid": ["alcohol", "fried food"]
    }
}

@app.route("/", methods=["GET", "POST"])
def index():
    bmi = diet = None
    foods = []
    allowed = []
    avoid = []

    if request.method == "POST":
        weight = float(request.form["weight"])
        height = float(request.form["height"]) / 100
        exercise = float(request.form["exercise"])
        disease = request.form["disease"]

        bmi = round(weight / (height ** 2), 2)

        # ML Prediction
        diet = model.predict([[bmi, exercise]])[0]

        # Food recommendation
        if diet == "Weight Loss":
            foods = food_df[food_df["calories"] < 150]["food_name"].head(5).tolist()
        elif diet == "Weight Gain":
            foods = food_df[food_df["calories"] > 300]["food_name"].head(5).tolist()
        else:
            foods = food_df["food_name"].head(5).tolist()

        if disease in disease_rules:
            allowed = disease_rules[disease]["allowed"]
            avoid = disease_rules[disease]["avoid"]

    return render_template(
        "index.html",
        bmi=bmi,
        diet=diet,
        foods=foods,
        allowed=allowed,
        avoid=avoid
    )

if __name__ == "__main__":
    app.run(debug=True)
