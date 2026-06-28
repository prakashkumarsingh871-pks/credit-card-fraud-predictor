from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import xgboost as xgb
from pydantic import BaseModel

app = FastAPI(title="Credit Card Fraud Predictor API")

# Enable CORS so your local HTML file can communicate with the local API server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define data schema using Pydantic for validation
class TransactionInput(BaseModel):
    customer_id: str
    age: int
    income: float
    credit_score: int
    credit_utilization: float
    missing_payment: int
    delinquent_account: int
    loan_balance: float
    debt_to_income: float
    employment_status: int  # Example encoding: 0=Unemployed, 1=Self-Employed, 2=Employed
    account_tenure: int
    credit_card_type: int  # Example encoding: 0=Standard, 1=Silver, 2=Gold, 3=Platinum
    month_1: float
    month_2: float
    month_3: float
    month_4: float
    month_5: float
    month_6: float
    income_imputed_flag: int  # 0=Actual, 1=Imputed


# Load saved XGBoost model
model = xgb.XGBClassifier()
try:
    model.load_model("fraud_model.json")
except Exception:
    print(
        "Warning: 'fraud_model.json' not found. Please run train_model.py first."
    )


@app.post("/predict")
def predict_fraud(data: TransactionInput):
    # Convert incoming input into a structural dictionary
    input_dict = {
        "age": [data.age],
        "income": [data.income],
        "credit_score": [data.credit_score],
        "credit_utilization": [data.credit_utilization],
        "missing_payment": [data.missing_payment],
        "delinquent_account": [data.delinquent_account],
        "loan_balance": [data.loan_balance],
        "debt_to_income": [data.debt_to_income],
        "employment_status": [data.employment_status],
        "account_tenure": [data.account_tenure],
        "credit_card_type": [data.credit_card_type],
        "month_1": [data.month_1],
        "month_2": [data.month_2],
        "month_3": [data.month_3],
        "month_4": [data.month_4],
        "month_5": [data.month_5],
        "month_6": [data.month_6],
        "income_imputed_flag": [data.income_imputed_flag],
    }

    # Format into a Pandas DataFrame to preserve column structures
    features_df = pd.DataFrame(input_dict)

    # Generate probabilities
    probabilities = model.predict_proba(features_df)
    fraud_probability = float(probabilities[0][1])

    # Fraud Status Threshold Logic
    if fraud_probability >= 0.75:
        verdict = "High Risk - FLAG & BLOCK"
    elif fraud_probability >= 0.40:
        verdict = "Medium Risk - REVIEW REQUIRED"
    else:
        verdict = "Low Risk - APPROVE"

    return {
        "customer_id": data.customer_id,
        "fraud_probability": round(fraud_probability * 100, 2),
        "verdict": verdict,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)