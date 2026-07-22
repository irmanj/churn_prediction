import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import joblib

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# print(df.head())
# print(df.info())
# print(df.describe(include="all"))

# hapus customerID
df = df.drop("customerID", axis=1)

# ubah TotalCharges menjadi numerik
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# cek missing value
print(df.isnull().sum())

# hapus baris yang memiliki missing value
df = df.dropna()

# print(df.shape)

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# print(df["Churn"].value_counts())

# menguubah semua data kategori menjadi angka
df = pd.get_dummies(df, drop_first=True)

# print(df.head())
# print(df.shape)

X = df.drop("Churn", axis=1)
y = df["Churn"]

# print(X.shape)
# print(y.shape)

# X = df[["umur", "langganan", "usage"]]
# y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# print(X_train.shape)
# print(X_test.shape)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# model = LogisticRegression(max_iter=1000)

# model = DecisionTreeClassifier(
#     random_state=42,
#     max_depth=5
# )

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfussion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, "models/churn_model.pkl")
joblib.dump(X.columns.tolist(), "models/feature_names.pkl")

print("Model saved!")