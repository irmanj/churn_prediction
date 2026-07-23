from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Customer Churn Prediction API")

# load model
model = joblib.load("models/churn_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

class Customer(BaseModel):
    SeniorCitizen: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

    gender_Male: int
    Partner_Yes: int
    Dependents_Yes: int
    PhoneService_Yes: int
    MultipleLines_Yes: int
    InternetService_Fiber_optic: int
    OnlineSecurity_Yes: int
    OnlineBackup_Yes: int
    DeviceProtection_Yes: int
    TechSupport_Yes: int
    StreamingTV_Yes: int
    StreamingMovies_Yes: int
    Contract_One_year: int
    Contract_Two_year: int
    PaperlessBilling_Yes: int
    PaymentMethod_Credit_card_automatic: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API"}

@app.post("/predict")
def predict(customer: Customer):

    sample = pd.DataFrame(0, index=[0], columns=feature_names)

    data = customer.model_dump()

    #Mapping nama field API -> nama kolom hasil get_dummies
    mapping = {
        "InternetService_Fiber_optic": "InternetService_Fiber optic",
        "Contract_One_year": "Contract_One year",
        "Contract_Two_year": "Contract_Two year",
        "PaymentMethod_Credit_card_automatic": "PaymentMethod_Credit card (automatic)",
        "PaymentMethod_Electronic_check": "PaymentMethod_Electronic check",
        "PaymentMethod_Mailed_check": "PaymentMethod_Mailed check",
    }

    for key, value in data.items():
        column = mapping.get(key, key)
        if column in sample.columns:
            sample.loc[0, column] = value

    prediction = int(model.predict(sample)[0])

    return {
        "prediction": prediction,
        "result": "Churn" if prediction == 1 else "Stay"
    }