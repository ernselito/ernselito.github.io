---
layout: single
title: "Modernizing Credit Risk: A Feature Engineering Approach"
excerpt: "A walkthrough Feature Engineering, and how I improved accuracy using feature engineering."
date: 2025-11-10
read_time: true
comments: true
share: true
related: true
header:
  overlay_color: "#000"
  overlay_filter: "0.5"
  overlay_image: "/assets/images/ml-project-banner.jpg"
  caption: "Exploring Feature Engineering"
class: wide
---


# Modernizing Credit Risk: A Feature Engineering Approach

**Focus:** Financial Risk Management & ML Strategy  
**Stack:** Python, XGBoost, Pandas, Scikit-Learn

### The Context

In lending, the difference between a profitable portfolio and a defaulted one often lies in the gray areas. Traditional credit models rely heavily on static snapshots—like raw income or credit score—which can miss the nuance of a borrower's actual real-time capacity to pay.

For this project, I built a machine learning pipeline designed to mimic the intuition of a human underwriter but at the scale of an algorithm. My goal was to move beyond "black box" predictions and create a model rooted in explainable financial logic.

-----

### My Approach

#### 1\. Translating "Gut Instinct" into Code

Raw data rarely tells the whole story. A borrower might have a high income but be drowning in debt, or a low credit score but excellent recent payment history. I focused my effort on **Domain-Driven Feature Engineering** to capture these nuances:

  * **Defining "True" Affordability:** I didn't just look at income; I calculated `available_income` (post-debt cash flow) and `payment_to_income` ratios. This reveals how tight the borrower’s budget actually is once they take on the new loan.
  * **Composite Risk Scoring:** I created a synthetic `default_risk` index—a weighted combination of their credit history, interest volatility, and debt burden—to give the model a single, clear signal of borrower health.

#### 2\. Protecting Against Bias & Leakage

Financial data is messy. Employment titles and sub-grades are high-cardinality categorical variables that can easily confuse a model.

  * **Target Encoding:** I implemented a K-Fold Target Encoding strategy. This allowed the model to learn the historical risk associated with specific job titles or grades without "peeking" at the answer key (data leakage), ensuring the model works in production, not just in testing.

#### 3\. Production-Ready Validation

A model is useless if it only works on one specific dataset.

  * **Stability First:** I used **8-Fold Stratified Cross-Validation**. By simulating performance across different random subsets of borrowers, I ensured the model remains stable even if the applicant pool fluctuates—a critical requirement for any production financial system.

-----

### The Code: Financial Logic to Python

The most impactful part of this project was the translation layer—turning financial concepts into vectorizable features. Here is the core logic I used to generate the risk metrics:

```python
def create_risk_features(df):
    """
    Transforms raw applicant data into actionable risk metrics
    mimicking underwriter logic.
    """
    df = df.copy()

    # 1. Real Affordability (Capacity to Pay)
    # Measures income remaining AFTER accounting for current debt load
    df["available_income"] = df["annual_income"] * (1 - df["debt_to_income_ratio"])
    
    # 2. Payment Strain
    # What % of monthly cash flow is eaten by THIS specific loan?
    df["monthly_payment"] = df["loan_amount"] * df["interest_rate"] / 1200
    df["payment_to_income"] = df["monthly_payment"] / (df["annual_income"] / 12 + 1)

    # 3. Composite Risk Index
    # A weighted score combining DTI, Credit Score, and Loan Cost.
    # We weight DTI highest (0.40) as it is the leading indicator of distress.
    df["composite_risk_score"] = (
        df["debt_to_income_ratio"] * 0.40
        + (850 - df["credit_score"]) / 850 * 0.35
        + df["interest_rate"] / 100 * 0.25
    )

    # 4. Credit Utilization Proxy
    # Helps differentiate high-income/high-debt from low-income/low-debt applicants
    df["credit_utilization_proxy"] = df["credit_score"] * (1 - df["debt_to_income_ratio"])

    return df
```

### The "So What?" (Business Impact)

By focusing on **Feature Engineering** rather than just hyperparameter tuning, this project achieved three key business outcomes:

1.  **Lower Default Exposure:** The `available_income` metric successfully flagged borrowers who looked good on paper (high income) but were actually over-leveraged.
2.  **Explainability:** Because the features are based on real financial concepts (like "Payment Strain"), the model's decisions can be explained to compliance teams and customers.
3.  **Scalability:** Using XGBoost with GPU acceleration allows this logic to be applied to thousands of applications in milliseconds, enabling real-time decisioning.