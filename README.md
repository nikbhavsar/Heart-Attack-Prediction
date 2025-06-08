# Capstone Project: Predicting Heart Attack Risk 
Author: Nikhar Bhavsar

**Date:** 2025/02/02

**Dataset link:** https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

## 1. Project Overview
### Problem Area

#### The Growing Crisis of Heart Disease
Heart disease is a leading cause of death in Canada, yet **1 in 3 cases go undiagnosed** until severe symptoms emerge. Delayed detection often leads to irreversible damage, increased healthcare costs, and preventable fatalities. With **14 Canadian adults dying every hour** from heart disease (CCDSS 2017–2018), early risk assessment is critical.

#### Barriers to Care
- **Doctor shortages**: 4.7 million Canadians lack a family physician ([2023 Statistics Canada](https://www.statcan.gc.ca)).
- **Limited awareness**: Many dismiss early warning signs like high blood pressure or cholesterol.
- **Testing gaps**: Rural/remote communities often lack diagnostic facilities.

### Proposed Data Science Solution
Machine learning can look for patterns in health data and predict if someone is at risk of heart disease. For example, based on data like high glucose or cholesterol
levels, it can show if the user might need medical attention.

## 2. Data Information

### Data Dictionary For This Project:

| **Feature Name**               | **Description**                                                                                     | **Data Type** |
|---------------------------------|-----------------------------------------------------------------------------------------------------|---------------|
| State                          | State FIPS Code                                                                                     | `object`      |
| Sex                            | Sex of Respondent                                                                                   | `object`      |
| GeneralHealth                  | Self-reported general health status                                                                 | `object`      |
| PhysicalActivities             | Participation in physical activities/exercises in past month                                        | `object`      |
| SleepHours                     | Average hours of sleep per 24-hour period                                                           | `float64`     |
| RemovedTeeth                   | Number of permanent teeth removed due to decay/gum disease                                          | `float64`     |
| HadHeartAttack                 | Ever diagnosed with heart attack                                                                    | `object`      |
| HadAngina                      | Ever diagnosed with angina/coronary heart disease                                                   | `object`      |
| HadStroke                      | Ever diagnosed with stroke                                                                          | `object`      |
| HadCOPD                        | Ever diagnosed with COPD/emphysema/chronic bronchitis                                               | `object`      |
| HadArthritis                   | Ever diagnosed with arthritis/rheumatoid arthritis/gout/lupus/fibromyalgia                          | `object`      |
| HadDiabetes                    | Ever diagnosed with diabetes                                                                        | `object`      |
| SmokerStatus                   | Smoking status (Everyday/Someday/Former/Non-smoker)                                                 | `object`      |
| ChestScan                      | Ever had CT/CAT scan of chest area                                                                  | `object`      |
| RaceEthnicityCategory          | Race/ethnicity classification                                                                       | `object`      |
| AgeCategory                    | Age group classification                                                                            | `object`      |
| HeightInMeters                 | Self-reported height in meters                                                                      | `float64`     |
| WeightInKilograms              | Self-reported weight in kilograms                                                                   | `float64`     |
| BMI                            | Calculated Body Mass Index                                                                          | `float64`     |
| AlcoholDrinkers                | Alcohol consumption in past 30 days                                                                 | `object`      |
| HighRiskLastYear               | High-risk behaviors in past year                                                                    | `object`      |

## 3. Project Workflow

### 1. Exploratory Data Analysis (EDA) and Data Cleaning
- Visualized feature distributions and relationships using graphs and plots.
- Look into the dataset for inconsistencies and missing values.
- Removed null values and handled outliers appropriately.

### 2. Data Preprocessing
- Encoded categorical variables using **OneHotEncoder** and **OrdinalEncoder**.
- Scaled numerical variables to ensure consistent ranges across features.

### 3. Hypothesis Testing
- Conducted hypothesis testing to validate assumptions.

### 4. Baseline Model
- Built a **Logistic Regression** model as the baseline for initial performance evaluation.

### 5. Model Development & Imbalance Handling
- Resampled the data using:
  - **SMOTEENN**
  - **SMOTENearestNeighbors**
- Tuned hyperparameters using **GridSearchCV** on:
  - Random Forest
  - Logistic Regression
  - XGBoost
  - LightGBM
- Compared models using:
  - Recall
  - F1 Score
  - Precision
  - ROC AUC / PR AUC

### 6. Final Model Training
- Selected **Random Forest** as the final model.
- Trained on the full data using optimal parameters.
- Saved model and feature importance:
  - `random_forest_heart_attack_model.pkl`
  - `feature_importance.csv`

### 7. Web App Deployment
- Built a web app with **Flask**
- Hosted at:  
  🔗 [https://predict-heart-attack-risk.onrender.com/](https://predict-heart-attack-risk.onrender.com/)
- Allows users to input key health indicators and receive risk level prediction: **Low**, **Moderate**, or **High**

---

## 4. Model Performance

| Metric            | Value      |
|-------------------|------------|
| Accuracy          | 85.9%      |
| Recall (Positive) | 70.5%      |
| Precision         | 23.6%      |
| F1 Score          | 0.354      |
| ROC AUC           | 0.879      |
| PR AUC            | 0.387      |

---

## 5. Outputs

- `random_forest_heart_attack_model.pkl` — Final model file
- `feature_importance.csv` — Top contributing features
- Precision-Recall Curve — Visual comparison of models
- Full classification report (see Notebook 6)

## 6. Limitations

- **Low Precision**: Although recall is high (to reduce false negatives), precision remains relatively low, which can lead to false alarms in a real-world scenario.
- **Imbalanced Dataset**: Despite using resampling techniques, the original class imbalance (~5% positive class) still limits precision and stability.