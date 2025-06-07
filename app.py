from flask import Flask, request, render_template
import joblib
import pandas as pd
from flask_cors import CORS
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Initialize the app
app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load("./models/random_forest_heart_attack_model.pkl")

# Define features used (based on importance)
used_features = [
    'HadAngina', 'ChestScan', 'AgeCategory', 'HadDiabetes', 'HadArthritis',
    'HadStroke', 'GeneralHealth', 'AlcoholDrinkers', 'HadCOPD', 'PhysicalActivities',
    'SmokerStatus', 'HadKidneyDisease', 'RaceEthnicityCategory',
    'Sex', 'SleepHours', 'HeightInMeters', 'WeightInKilograms'
]

# Map features to user-friendly labels
feature_labels = {
    'HadAngina': 'Had Angina?',
    'ChestScan': 'Had Chest Scan?',
    'AgeCategory': 'Age Category (e.g., 55-64)',
    'HadDiabetes': 'Diagnosed with Diabetes?',
    'HadArthritis': 'Diagnosed with Arthritis?',
    'HadStroke': 'Previously had a Stroke?',
    'GeneralHealth': 'General Health (e.g., Excellent, Fair)',
    'AlcoholDrinkers': 'Consumes Alcohol?',
    'HadCOPD': 'Has COPD?',
    'PhysicalActivities': 'Physically Active?',
    'SmokerStatus': 'Smoking Status (Never, Former, Current)',
    'HadKidneyDisease': 'Diagnosed with Kidney Disease?',
    'RaceEthnicityCategory': 'Race/Ethnicity (e.g., White, Hispanic, Other)',
    'Sex': 'Sex (Male or Female)',
    'SleepHours': 'Average Sleep Hours (per night)',
    'HeightInMeters': 'Height (in meters)',
    'WeightInKilograms': 'Weight (in kilograms)'
}

@app.route("/", methods=["GET"])
def home():
    return render_template("./templates/index.html", used_features=used_features, feature_labels=feature_labels)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form
        input_data = {}

        for feature in used_features:
            val = data.get(feature)
            if val is None:
                raise ValueError(f"Missing input for: {feature}")

            # Convert feature types
            if feature in ['HadAngina', 'HadDiabetes', 'HadArthritis', 'HadStroke', 'AlcoholDrinkers', 'HadCOPD', 'HadKidneyDisease', 'PhysicalActivities']:
                input_data[feature] = val.lower() == 'yes'
            elif feature in ['SleepHours', 'HeightInMeters', 'WeightInKilograms']:
                input_data[feature] = float(val)
            elif feature in ['Sex']:
                input_data['Sex_Female'] = int(val == 'Female')
                input_data['Sex_Male'] = int(val == 'Male')
            elif feature == 'SmokerStatus':
                input_data['SmokerStatus_Never smoked'] = int(val == 'Never')
                input_data['SmokerStatus_Former smoker'] = int(val == 'Former')
            elif feature == 'RaceEthnicityCategory':
                input_data['RaceEthnicityCategory_White'] = int(val == 'White')
                input_data['RaceEthnicityCategory_Hispanic'] = int(val == 'Hispanic')
            else:
                input_data[feature] = val

        df = pd.DataFrame([input_data])
        prob = model.predict_proba(df)[0][1]
        prediction = int(prob >= 0.35)
        risk_level = "High Risk" if prediction == 1 else "Low Risk"

        return render_template("./templates/index.html", result=risk_level, prob=round(prob * 100, 2), used_features=used_features, feature_labels=feature_labels)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
