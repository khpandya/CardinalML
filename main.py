from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Define a class for the input data model
class InputFeatures(BaseModel):
    Age: int
    Sex: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str


# Initialize the FastAPI app
app = FastAPI()
# Load the trained model
model = joblib.load("./random_forest_heart_disease_classifier.pkl")
# Load the SHAP Explainer
explainer = joblib.load("./explainer.pkl")
preprocessor = model.named_steps['preprocessor']
feature_names = joblib.load("./feature_names.pkl")

@app.post("/predict")
def predict(input_data: InputFeatures):
    try:
        # Convert input data to a dataframe (as expected by the model)
        input_df = pd.DataFrame([input_data.dict()])

        # Get the probability of HeartDisease being 1
        probability = model.predict_proba(input_df)[0][1] * 100  # Multiply by 100 for percentage

        # preprocess data to convert it to the format the explainer expects
        preprocessed_input_df = preprocessor.transform(input_df)
        # Calculate SHAP values for the preprocessed input data
        shap_values = explainer.shap_values(preprocessed_input_df, check_additivity=False)

        # Mapping feature names to their corresponding SHAP values
        feature_shap_map = {feature_names[i]: shap_values[1][0][i] for i in range(len(feature_names))}
        return {
            "HeartDiseaseProbability": f"{probability:.2f}%",
            "SHAPValues": feature_shap_map
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run this script directly to start the server: uvicorn script_name:app --reload


# fastapi==0.75.1
# uvicorn==0.17.6
# joblib==1.1.0
# pandas==1.3.5
# needs sklearn too
# scikit-learn==1.0.2