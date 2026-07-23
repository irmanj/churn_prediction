# Customer Churn Prediction

## Project Overview

Machine Learning project to predict whether a customer will churn using Logistic Regression.

## Dataset

IBM Telco Customer Churn Dataset

## Features

- tenure
- MonthlyCharges
- TotalCharges
- InternetService
- Contract
- PaymentMethod
- etc

## Model

- Logistic Regression

Accuracy: 80.31%

## API

POST /predict

## Tech Stack

- Python
- Pandas
- Scikit-learn
- FastAPI
- Joblib

## Run

pip install -r requirements.txt

uvicorn src.app:app --reload