# Customer Churn Prediction

## Project Overview

Customer churn is a major challenge for subscription-based companies. 
The objective of this project is to analyze customer characteristics and 
build a machine learning model to identify customers who are likely to churn.

This project covers the complete data science workflow, from exploratory 
data analysis to model interpretation and business insights.

## Objectives

- Explore customer characteristics and identify patterns associated with churn.
- Analyze the relationship between customer behavior and churn.
- Preprocess numerical and categorical features.
- Build and compare machine learning models.
- Address class imbalance.
- Interpret the final Logistic Regression model.
- Translate model results into actionable business insights.

## Dataset

The dataset contains **7,043 customers** and **21 variables**, including:

- Customer demographics
- Contract information
- Internet and phone services
- Payment methods
- Monthly and total charges
- Customer tenure
- Churn status

The target variable is `Churn`, indicating whether a customer left the company.

### Target distribution

- **No Churn:** 73.46%
- **Churn:** 26.54%

This class imbalance is taken into account during the modeling process.

## Exploratory Data Analysis

The exploratory analysis investigates the relationship between churn and:

- Contract type
- Tenure
- Internet service
- Payment method
- Monthly charges
- Total charges
- Technical support and online security
- Demographic and service-related features

Several patterns were observed. Churn is notably higher among customers with:

- Month-to-month contracts
- Shorter tenure
- Fiber optic internet service
- Electronic check payment
- No online security or technical support

These observations describe associations in the dataset and should not be interpreted as causal relationships.

## Data Preprocessing

The preprocessing pipeline includes:

- Conversion of `TotalCharges` to numerical values
- Handling of missing values
- Separation of numerical and categorical features
- Standardization of numerical variables
- One-hot encoding of categorical variables
- Stratified train/test split

The data was split into:

- **80% training set:** 5,634 customers
- **20% test set:** 1,409 customers

After preprocessing, the feature matrix contains **45 features**.

## Machine Learning

Two initial models were evaluated:

- Logistic Regression
- Random Forest

A majority-class baseline was also used for comparison.

### Baseline

The baseline accuracy is:

**73.46%**

### Model comparison

| Model | Accuracy | Churn Precision | Churn Recall | Churn F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.55% | 65.72% | 55.88% | 60.40% | 0.842 |
| Random Forest | 77.50% | 60.00% | 47.00% | 53.00% | 0.819 |

## Handling Class Imbalance

Because churn represents only 26.54% of the dataset, accuracy alone is not sufficient to evaluate the model.

A balanced Logistic Regression model was therefore trained using:

```python
class_weight="balanced"

### Balanced Logistic Regression Results

The balanced Logistic Regression model achieved:

| Metric | Result |
|---|---:|
| Accuracy | 73.81% |
| Precision (Churn) | 50.43% |
| Recall (Churn) | 78.34% |
| F1-score (Churn) | 61.36% |
| ROC-AUC | 0.842 |

Compared with the original Logistic Regression model, the balanced model increased recall for churned customers from 55.88% to 78.34%.

This means that the model identifies more potential churners, but also produces more false positives. This trade-off is important when the objective is to detect as many potential churners as possible.
