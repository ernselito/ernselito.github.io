---
layout: single
title: "AI-Driven Credit Risk Engine"
date: 2025-11-24
read_time: true
comments: true
share: true
related: true
class: wide
---

**Role:** Machine Learning Engineer

**Tech:** Python, XGBoost, Scikit-Learn

**Outcome:** Increased fraud detection by tuning decision thresholds, achieving 0.89 AUC.

## 1. The Business Challenge

Traditional credit scoring relies on static metrics (like FICO scores) that often miss a borrower's real-time financial health. A borrower might have a good credit score but be over-leveraged with monthly payments, creating a "hidden" default risk.

Objective: Build a machine learning model that looks beyond the credit score to assess "True Affordability," reducing financial exposure to bad loans without rejecting valid customers.

## 2. The Solution: Behavioral Feature Engineering

Instead of feeding raw data into a model, I engineered features that mimic the logic of a human underwriter:

-**Payment Strain:** Calculated the ratio of the new loan payment to monthly income.

-**Free Cash Flow:** Modeled available_income after existing debts are paid.

I trained an XGBoost model on these engineered features using 8-Fold Stratified Cross-Validation to ensure stability across different customer segments.

## 3. The "Aha!" Moment (Optimization)

The initial model had a high accuracy but approved too many risky loans (False Positives) because it used a standard 50% probability threshold.

-**The Fix:** I performed Threshold Tuning, shifting the approval boundary to 65% probability.

-**The Impact:** This stricter standard prioritized capital preservation. While it slightly reduced the overall approval volume, it drastically cut the number of expected defaults, directly protecting the bank's bottom line.

## 4. Key Results

-**Performance:** Achieved an ROC-AUC score of 0.893, indicating excellent discriminatory power between payers and defaulters.

-**Impact:** Engineered features like payment_strain consistently ranked in the top 3 predictors, proving that behavioral metrics are more predictive than demographic data.

Code snapshot below.

```python
def make_credit_decision(applicant_data, model):
    """
    Accepts a raw applicant dictionary, engineers features in real-time,
    and returns a business decision (Approve/Decline).
    """
    # 1. Real-time Feature Engineering
    income = applicant_data['annual_income']
    monthly_payment = (applicant_data['loan_amount'] * (applicant_data['interest_rate'] / 100)) / 12
    
    # 2. Key Metric: Payment Strain
    payment_strain = monthly_payment / (income / 12)
    
    # 3. Prediction
    # We use a custom 0.65 threshold to prioritize safety
    prob_repayment = model.predict_proba(processed_data)[0][1]
    decision = "APPROVE" if prob_repayment > 0.65 else "DECLINE"
    
    return decision, payment_strain
```
