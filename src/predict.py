import joblib
import pandas as pd

# Load model dan feature names
model = joblib.load("models/churn_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# Data kosong
sample = pd.DataFrame(0, index=[0], columns=feature_names)

# Isi data numerik
sample.loc[0, "SeniorCitizen"] = 0
sample.loc[0, "tenure"] = 12
sample.loc[0, "MonthlyCharges"] = 75.5
sample.loc[0, "TotalCharges"] = 906

# Isi data kategori (hasil one-hot encoding)
sample.loc[0, "gender_Male"] = 1
sample.loc[0, "Partner_Yes"] = 1
sample.loc[0, "Dependents_Yes"] = 0
sample.loc[0, "PhoneService_Yes"] = 1
sample.loc[0, "MultipleLines_Yes"] = 1
sample.loc[0, "InternetService_Fiber optic"] = 1
sample.loc[0, "OnlineSecurity_Yes"] = 1
sample.loc[0, "OnlineBackup_Yes"] = 1
sample.loc[0, "DeviceProtection_Yes"] = 1
sample.loc[0, "TechSupport_Yes"] = 0
sample.loc[0, "StreamingTV_Yes"] = 1
sample.loc[0, "StreamingMovies_Yes"] = 1
sample.loc[0, "Contract_One year"] = 1
sample.loc[0, "PaperlessBilling_Yes"] = 1
sample.loc[0, "PaymentMethod_Electronic check"] = 1

prediction = model.predict(sample)

print(prediction)

if prediction[0] == 1:
    print("⚠️ Customer will Churn")
else:
    print("✅ Customer will Stay")