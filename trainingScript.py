## NOT PART OF THIS PROJECT
## THIS SCRIPT MAY NOT WORK CORRECTLY IN THIS ENVIRONMENT AND IS JUST FOR REFERENCE
## requirements.txt pasted at the end, advisable to move it to another project
## heart.csv included in the project for reference

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import joblib
import shap

# Load the dataset
file_path = './heart.csv'
data = pd.read_csv(file_path)

# Identify categorical and numerical columns
categorical_cols = ['Sex', 'RestingECG', 'ExerciseAngina']
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR']

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Create the Random Forest Classifier
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier())])

# Splitting the dataset into training and testing sets
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model.fit(X_train, y_train)

# Evaluating the model
predictions = model.predict(X_test)
report = classification_report(y_test, predictions)
print(predictions)
print(report)

# Save the model to a file
model_filename = './random_forest_heart_disease_classifier.pkl'
joblib.dump(model, model_filename)

# Save preprocessed feature names (important for mapping SHAP values)
# This includes names of the one-hot encoded features
feature_names = (numerical_cols +
                 list(model.named_steps['preprocessor']
                      .named_transformers_['cat']
                      .get_feature_names_out(categorical_cols)))
feature_names_filename = './feature_names.pkl'
joblib.dump(feature_names, feature_names_filename)

# Create a SHAP Explainer
explainer = shap.Explainer(model.named_steps['classifier'], model.named_steps['preprocessor'].transform(X_train))

# Save the SHAP Explainer to a file
explainer_filename = './explainer.pkl'
joblib.dump(explainer, explainer_filename)

# annotated-types==0.6.0
# anyio==4.2.0
# click==8.1.7
# cloudpickle==3.0.0
# colorama==0.4.6
# exceptiongroup==1.2.0
# fastapi==0.109.0
# h11==0.14.0
# httptools==0.6.1
# idna==3.6
# importlib-metadata==7.0.1
# joblib==1.3.2
# llvmlite==0.41.1
# numba==0.58.1
# numpy==1.24.4
# packaging==23.2
# pandas==2.0.3
# pydantic==2.5.3
# pydantic_core==2.14.6
# python-dateutil==2.8.2
# python-dotenv==1.0.1
# pytz==2023.4
# PyYAML==6.0.1
# scikit-learn==1.0.2
# scipy==1.10.1
# shap==0.44.1
# six==1.16.0
# slicer==0.0.7
# sniffio==1.3.0
# starlette==0.35.1
# threadpoolctl==3.2.0
# tqdm==4.66.1
# typing_extensions==4.9.0
# tzdata==2023.4
# uvicorn==0.27.0
# watchfiles==0.21.0
# websockets==12.0
# zipp==3.17.0

