---
layout: single
title: "Feature Engineering: The Secret Weapon That Decoded California Housing Prices"
excerpt: "A walkthrough Feature Engineering, and how I improved accuracy using feature engineering."
date: 2025-11-10
read_time: true
comments: true
share: true
related: true
class: wide
---


# Feature Engineering: The Secret Weapon That Decoded California Housing Prices

[View Full Code On Github](https://github.com/ernselito/Impact-of-Feature-Transformation-on-Machine-Learning-)

### The Problem with Raw Data
Accurate housing price prediction is the holy grail for real estate investors and market analysts. But here’s the secret: raw data is rarely good enough.

When I started my project to predict California housing prices, my initial models (the baseline) performed reasonably well. However, I knew the raw dataset was missing crucial, common-sense factors. For example, the data had total_rooms and households, but no measure for the average number of rooms per family—a feature that intuitively impacts property value.

This led to my core research question: How much does intelligent feature engineering actually improve the performance of machine learning models in predicting California housing prices?

### Experiment 1: The Baseline Challenge
My first step was establishing a baseline performance using common preprocessing (imputation and scaling) but no feature creation. I tested six standard regression models on the California Housing dataset:


| Model | Baseline RMSE (USD) |
| --- | --- |
| Random Forest | 53,747 |
| Gradient Boosting | 55,486 |
| K-Nearest Neighbors | 60,685 |
| Linear Regression | 67,354 |
| Decision Tree | 75,036 |
| SVR | 116,889 |


The Random Forest Regressor was the clear winner, achieving an RMSE (Root Mean Squared Error) of approximately \$ 53,747. This means, on average, the model's prediction was off by about \$ 53,747. This is a decent starting point, but a margin we aimed to shrink.

### Experiment 2: Creating Basic Ratio Features

Next, I focused on creating new, meaningful features by combining existing columns. These are features a human expert would instinctively use to evaluate a property:

- rooms_per_household: total_rooms / households (Size proxy)

- bedrooms_per_room: total_bedrooms / total_rooms (Quality/Density proxy)

- population_per_household: population / households (Density proxy)


| Model | Baseline RMSE (USD) | Basic Features RMSE (USD) | Change |
| --- | --- | --- | --- |
| Random Forest | 53,747 | 53,253 | -0.92% |
| Gradient Boosting | 55,486 | 54,833 | -1.18% |
| K-Nearest Neighbors | 60,685 | 66,817 | +10.09% |


Simply adding these three ratio features slightly improved the RMSE for the top models (Random Forest and Gradient Boosting). However, they worsened the performance of the K-Nearest Neighbors model, demonstrating that not all feature engineering benefits all model types equally. The Random Forest model's RMSE dropped to $53,253.

### Experiment 3: Advanced Spatial Features

To capture geographic factors, which are paramount in real estate, I engineered advanced features: spatial proximity. I calculated the Haversine distance (actual distance over the earth's surface) from each house to major economic hubs: Los Angeles and San Francisco.

```python
def transform(self, X, y=None):
        X = X.copy()

        # Calculate new features
        X['rooms_per_household'] = X['total_rooms'] / X['households']
        X['bedrooms_per_room'] = X['total_bedrooms'] / X['total_rooms']
        X['population_per_household'] = X['population'] / X['households']

        # Calculate distance to each city
        for city, (lon, lat) in self.cities.items():
            X[f'distance_to_{city}'] = self.haversine_distance(X, lon, lat)

        return X
```

This new feature set included the three ratios from Experiment 2 plus the two new distance metrics.


| Model | Basic Features RMSE (USD) | Advanced Features RMSE (USD) | Change |
| --- | --- | --- | --- |
| Random Forest | 53,253 | 54,652 | +2.62% |
| Gradient Boosting | 54,833 | 54,396 | -0.79% |

Advanced Feature Insight

Interestingly, the Gradient Boosting Regressor took the lead in this phase, achieving a minor but measurable drop to $54,396. While the Random Forest Regressor's score worsened, the overall best performing model now included the spatial features, suggesting these features captured variance that boosted the Gradient Boosting model's predictive power.

## The Final Leap: Hyperparameter Tuning

To unlock the full potential of feature engineering, I took the best model—the Random Forest Regressor (which performed best across all feature sets post-tuning)—and fine-tuned its hyperparameters using GridSearchCV.

I performed this tuning across all three feature sets (Baseline, Basic, Advanced) to see which combination yielded the absolute lowest error.


| Feature Set | Best Post-Tuning RMSE (USD) | Best Model Parameters | Total Improvement (vs. Baseline) |
| --- | --- | --- | --- |
| Baseline | 49,659 | max_features=8, n_estimators=30 | 7.6% |
| Basic Ratios | 49,763 | max_features=8, n_estimators=30 | 7.4% |
| Advanced (Ratios + Distance) | 47,231 | max_features=6, n_estimators=30 | 12.1% |


### Key Takeaway

The combination of Advanced Feature Engineering (Ratios + Distance) and Hyperparameter Tuning delivered the absolute best result: an RMSE of $47,231. This represents a 12.1% reduction in prediction error compared to the initial baseline model.

Conclusion: Feature Engineering is Not Optional

This project conclusively proves that feature engineering is critical for developing high-accuracy predictive models in real estate. The creation of ratio-based features (like rooms per household) and spatial features (distance to cities) gave the model meaningful context that the raw data lacked.

The Random Forest Regressor, when combined with the engineered features and proper tuning, provided the most accurate and interpretable housing price predictions. The final model is off by less than $47,500 on average, demonstrating a significant improvement in reliability for informed decision-making.
