from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
from flask_cors import CORS
import sys
import os

sys.path.append(os.path.abspath('./utilities'))
import pre_processing_utils
import model_training_utils

app = Flask(__name__)
CORS(app)

# Load the pipeline
model_pipeline = joblib.load('./models/random_forest_heart_attack_model.pkl')

used_features = [
    'HadAngina', 'ChestScan', 'AgeCategory', 'HadDiabetes', 'HadArthritis',
    'HadStroke', 'GeneralHealth', 'AlcoholDrinkers', 'HadCOPD', 'PhysicalActivities',
    'SmokerStatus', 'HadKidneyDisease', 'RaceEthnicityCategory',
    'Sex', 'SleepHours', 'HeightInMeters', 'WeightInKilograms'
]

bool_cat_defaults = [
    'HadAngina', 'ChestScan', 'HadDiabetes', 'HadArthritis', 'HadStroke',
    'AlcoholDrinkers', 'HadCOPD', 'PhysicalActivities', 'HadKidneyDisease'
]

numeric_defaults = ['SleepHours', 'HeightInMeters', 'WeightInKilograms']

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", used_features=used_features)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()

        if not data:
            return jsonify({"error": "No input data received"}), 400

        # Default State
        if 'State' not in data or not data['State']:
            data['State'] = 'Alabama'

        # Default booleans/categoricals to 'No'
        for feature in bool_cat_defaults:
            if feature not in data or not data[feature]:
                data[feature] = 'No'

        # Default numerics to 0
        for feature in numeric_defaults:
            if feature not in data or not data[feature]:
                data[feature] = 0

        # Calculate BMI if possible
        try:
            height = float(data.get('HeightInMeters', 0))
            weight = float(data.get('WeightInKilograms', 0))
            data['BMI'] = round(weight / (height ** 2), 2) if height > 0 else 0
        except:
            data['BMI'] = 0

        all_features = used_features + ['State', 'BMI']
        input_dict = {f: data.get(f, None) for f in all_features}
        input_df = pd.DataFrame([input_dict])

        prob = model_pipeline.predict_proba(input_df)[0][1]
        # Assign risk category
        if prob < 0.30:
            risk_level = "Low Risk"
            message = "Great job! Keep maintaining your healthy habits."
        elif prob < 0.45:
            risk_level = "Moderate Risk"
            message = "Consider reviewing your health habits. Regular checkups can help."
        else:
            risk_level = "High Risk"
            message = "Consult your healthcare provider for professional advice."

        return jsonify({
            "risk": risk_level,
            "probability": round(prob * 100, 2),
            "message": message
        })
        print(prob)

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
