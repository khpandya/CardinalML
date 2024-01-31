import joblib
import pandas as pd

model = joblib.load("./random_forest_heart_disease_classifier.pkl")
preprocessor = model.named_steps['preprocessor']
# Load the SHAP Explainer
explainer = joblib.load("./explainer.pkl")
# Load the saved feature names
feature_names = joblib.load("./feature_names.pkl")

input_data = pd.DataFrame([{
    "Age": 20,
    "Sex": "M",
    "RestingBP": 100,
    "Cholesterol": 240,
    "FastingBS": 0,
    "RestingECG": "LVH",
    "MaxHR": 170,
    "ExerciseAngina": "Y"
}])

probability = model.predict_proba(input_data)[0][1] * 100  # Multiply by 100 for percentage
print("HeartDiseaseProbability:", probability)
# Preprocess the input data
preprocessed_input_df = preprocessor.transform(input_data)
# Calculate SHAP values for the preprocessed input data
shap_values = explainer.shap_values(preprocessed_input_df, check_additivity=False)

# Mapping feature names to their corresponding SHAP values
feature_shap_map = {feature_names[i]: shap_values[1][0][i] for i in range(len(feature_names))}
print(feature_shap_map)