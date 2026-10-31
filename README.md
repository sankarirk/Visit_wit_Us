
# Visit with Us – Wellness Tourism Package Purchase Prediction

## 1. Project Overview

This project develops an end-to-end Machine Learning and MLOps
solution to predict whether a customer is likely to purchase
a wellness tourism package.

The solution uses customer demographic information, travel
preferences, interaction details, and product-related features
to predict purchase behaviour.

The project demonstrates the complete machine learning lifecycle,
including data registration, data preparation, model experimentation,
model serialization, automated pipeline execution, and cloud deployment.

## 2. Business Objective

The primary objective is to help tourism businesses identify
customers who are more likely to purchase wellness tourism packages.

The prediction system can support:

- Customer targeting and segmentation.
- Marketing campaign optimization.
- Personalized tourism package recommendations.
- Improved sales conversion strategies.
- Data-driven business decisions.

## 3. Dataset Description

The project uses the tourism dataset containing 4,128 customer records
and 21 original columns.

Target variable:

- `ProdTaken`: Indicates whether the customer purchased the package.
- `0`: Did not purchase.
- `1`: Purchased.

The original dataset contains 18.69% positive-class records approximately
and 81.31% negative-class records approximately.

The columns `Unnamed: 0` and `CustomerID` are excluded from model training
because they are an index and customer identifier, respectively.

## 4. Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- MLflow
- Joblib
- Streamlit
- GitHub
- GitHub Actions

## 5. Project Structure

```text
Visit_wit_Us/
│
├── README.md
│
├── tourism_project/
│   ├── data/
│   │   └── tourism.csv
│   │
│   ├── model_building/
│   │   ├── data_register.py
│   │   ├── prep.py
│   │   └── train.py
│   │
│   ├── deployment/
│   │   ├── app.py
│   │   ├── best_model.joblib
│   │   ├── model_metadata.json
│   │   ├── model_comparison.csv
│   │   └── requirements.txt
│   │
│   └── requirements.txt
│
└── .github/
    └── workflows/
        └── pipeline.yml
```

## 6. Data Registration

The data registration module:

- Loads the tourism dataset.
- Validates expected dataset columns.
- Checks missing values.
- Identifies duplicate records.
- Examines the target variable distribution.

## 7. Data Preparation

The data preparation module:

- Removes unnecessary identifier columns.
- Separates independent and dependent variables.
- Performs a stratified training and testing split.
- Saves training and testing datasets for model development.

The dataset is divided into 80% training data and 20% testing data.

## 8. Model Building and Experiment Tracking

Multiple classification algorithms are evaluated:

- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting

Hyperparameter tuning is performed using GridSearchCV
with stratified cross-validation.

MLflow is used to track model parameters, metrics,
and experiment artifacts.

The best-performing model is selected using cross-validation F1-score.

### Selected Model

Random Forest Classifier

### Evaluation Results

| Metric | Result |
|---|---:|
| Accuracy | 90.80% |
| Precision | 84.30% |
| Recall | 64.15% |
| F1-score | 72.86% |
| ROC-AUC | 94.87% |

## 9. Model Deployment

The selected model is serialized using Joblib.

The Streamlit application loads the saved preprocessing
and classification pipeline and accepts customer information
through an interactive interface.

The application displays the predicted purchase outcome.

## 10. GitHub Actions MLOps Pipeline

The automated GitHub Actions workflow contains three jobs:

1. Register Dataset
2. Data Preparation
3. Model Training

The workflow automates dataset validation, preparation,
model training, experiment tracking, and model artifact updates.

## 11. Live Application

**Streamlit Application:**

https://visitwitus-j8nwkzweraatxfuwfyrnwa.streamlit.app/

## 12. GitHub Repository

**Repository:**

https://github.com/sankarirk/Visit_wit_Us

## 13. Conclusion

This project demonstrates an end-to-end MLOps implementation
for wellness tourism package purchase prediction.

It integrates data validation, preprocessing, model experimentation,
hyperparameter optimization, experiment tracking, model serialization,
continuous integration, and cloud deployment.

The solution provides a foundation for data-driven customer targeting
and tourism package marketing decisions.
