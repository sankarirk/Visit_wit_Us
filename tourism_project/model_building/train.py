
import os
import json
import warnings
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TARGET_COLUMN = "ProdTaken"

XTRAIN_PATH = "Xtrain.csv"
XTEST_PATH = "Xtest.csv"
YTRAIN_PATH = "ytrain.csv"
YTEST_PATH = "ytest.csv"

DEPLOYMENT_DIR = "tourism_project/deployment"
MODEL_PATH = os.path.join(
    DEPLOYMENT_DIR,
    "best_model.joblib"
)

os.makedirs(DEPLOYMENT_DIR, exist_ok=True)

print("=" * 75)
print("VISIT WITH US - MODEL TRAINING AND EXPERIMENT TRACKING")
print("=" * 75)

# ============================================================
# 2. LOAD TRAINING AND TESTING DATA
# ============================================================

required_files = [
    XTRAIN_PATH,
    XTEST_PATH,
    YTRAIN_PATH,
    YTEST_PATH
]

for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(
            f"Required dataset file not found: {file}"
        )

X_train = pd.read_csv(XTRAIN_PATH)
X_test = pd.read_csv(XTEST_PATH)

y_train = pd.read_csv(YTRAIN_PATH)[TARGET_COLUMN]
y_test = pd.read_csv(YTEST_PATH)[TARGET_COLUMN]

print("\nDATA LOADING")
print("-" * 50)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

if list(X_train.columns) != list(X_test.columns):
    raise ValueError(
        "Training and testing feature columns do not match."
    )

if len(X_train) != len(y_train):
    raise ValueError("Training feature and target sizes differ.")

if len(X_test) != len(y_test):
    raise ValueError("Testing feature and target sizes differ.")

if set(y_train.unique()) != {0, 1}:
    raise ValueError("Training target must contain classes 0 and 1.")

# ============================================================
# 3. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nFEATURE INFORMATION")
print("-" * 50)

print("Numerical features:", numeric_features)
print("Categorical features:", categorical_features)

# ============================================================
# 4. BUILD PREPROCESSING PIPELINE
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ],
    remainder="drop"
)

# ============================================================
# 5. DEFINE CLASSIFIERS AND HYPERPARAMETER GRIDS
# ============================================================

model_configurations = {
    "DecisionTree": {
        "classifier": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
        "parameters": {
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2]
        }
    },

    "RandomForest": {
        "classifier": RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1
        ),
        "parameters": {
            "classifier__n_estimators": [150, 250],
            "classifier__max_depth": [10, None],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2]
        }
    },

    "AdaBoost": {
        "classifier": AdaBoostClassifier(
            random_state=RANDOM_STATE
        ),
        "parameters": {
            "classifier__n_estimators": [50, 100],
            "classifier__learning_rate": [0.5, 1.0]
        }
    },

    "GradientBoosting": {
        "classifier": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
        "parameters": {
            "classifier__n_estimators": [100, 150],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [2, 3]
        }
    }
}

print("\nMODELS SELECTED FOR EXPERIMENTATION")
print("-" * 50)

for model_name in model_configurations:
    print("-", model_name)

# ============================================================
# 6. CONFIGURE MLFLOW EXPERIMENT TRACKING
# ============================================================

mlflow.set_tracking_uri("file:./mlruns")

experiment_name = "Visit_With_Us_Tourism_Prediction"

mlflow.set_experiment(experiment_name)

print("\nMLflow tracking URI:", mlflow.get_tracking_uri())
print("MLflow experiment:", experiment_name)

# ============================================================
# 7. HYPERPARAMETER TUNING AND MODEL EXPERIMENTATION
# ============================================================

cv_strategy = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

experiment_results = []

best_overall_score = -1
best_model = None
best_model_name = None
best_parameters = None

print("\nSTARTING MODEL EXPERIMENTS")
print("=" * 75)

for model_name, configuration in model_configurations.items():

    print(f"\nTraining and tuning: {model_name}")
    print("-" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", configuration["classifier"])
        ]
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=configuration["parameters"],
        scoring="f1",
        cv=cv_strategy,
        n_jobs=-1,
        refit=True,
        verbose=1,
        error_score="raise"
    )

    with mlflow.start_run(run_name=model_name):

        # Perform hyperparameter tuning
        grid_search.fit(X_train, y_train)

        tuned_model = grid_search.best_estimator_

        # Evaluate the tuned model on unseen test data
        y_pred = tuned_model.predict(X_test)

        y_probability = tuned_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            y_probability
        )

        cv_f1 = grid_search.best_score_

        # Log parameters and metrics in MLflow
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("cv_strategy", "StratifiedKFold_5")
        mlflow.log_param("scoring_metric", "f1")
        mlflow.log_params(grid_search.best_params_)

        mlflow.log_metric("best_cv_f1", float(cv_f1))
        mlflow.log_metric("test_accuracy", float(accuracy))
        mlflow.log_metric("test_precision", float(precision))
        mlflow.log_metric("test_recall", float(recall))
        mlflow.log_metric("test_f1", float(f1))
        mlflow.log_metric("test_roc_auc", float(roc_auc))

        # Store experiment results
        result = {
            "model": model_name,
            "best_cv_f1": float(cv_f1),
            "test_accuracy": float(accuracy),
            "test_precision": float(precision),
            "test_recall": float(recall),
            "test_f1": float(f1),
            "test_roc_auc": float(roc_auc),
            "best_parameters": grid_search.best_params_
        }

        experiment_results.append(result)

        # Log the tuned model in MLflow
        mlflow.sklearn.log_model(
            tuned_model,
            artifact_path="model"
        )

        print("\nBest hyperparameters:")
        print(grid_search.best_params_)

        print("\nEvaluation metrics:")
        print(f"Cross-validation F1: {cv_f1:.4f}")
        print(f"Test Accuracy:       {accuracy:.4f}")
        print(f"Test Precision:      {precision:.4f}")
        print(f"Test Recall:         {recall:.4f}")
        print(f"Test F1-score:       {f1:.4f}")
        print(f"Test ROC-AUC:        {roc_auc:.4f}")

        # Select the best model using cross-validation F1
        if cv_f1 > best_overall_score:
            best_overall_score = cv_f1
            best_model = tuned_model
            best_model_name = model_name
            best_parameters = grid_search.best_params_

# ============================================================
# 8. COMPARE EXPERIMENT RESULTS
# ============================================================

results_df = pd.DataFrame(experiment_results)

results_df = results_df.sort_values(
    by="best_cv_f1",
    ascending=False
).reset_index(drop=True)

print("\n\nMODEL COMPARISON RESULTS")
print("=" * 75)

print(
    results_df[
        [
            "model",
            "best_cv_f1",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_roc_auc"
        ]
    ].to_string(index=False)
)

# Save comparison results
results_path = os.path.join(
    DEPLOYMENT_DIR,
    "model_comparison.csv"
)

results_df.to_csv(results_path, index=False)

# ============================================================
# 9. FINAL EVALUATION OF THE SELECTED MODEL
# ============================================================

print("\n\nBEST MODEL SELECTED")
print("=" * 75)

print("Selected model:", best_model_name)
print("Best cross-validation F1-score:", best_overall_score)
print("Best hyperparameters:", best_parameters)

final_predictions = best_model.predict(X_test)

final_probabilities = best_model.predict_proba(X_test)[:, 1]

print("\nFINAL CLASSIFICATION REPORT")
print("-" * 60)

print(
    classification_report(
        y_test,
        final_predictions,
        target_names=[
            "Not Purchased",
            "Purchased"
        ],
        zero_division=0
    )
)

print("\nCONFUSION MATRIX")
print("-" * 60)

cm = confusion_matrix(y_test, final_predictions)

print(cm)

# ============================================================
# 10. SAVE THE BEST MODEL FOR DEPLOYMENT
# ============================================================

joblib.dump(best_model, MODEL_PATH)

print("\nMODEL SAVING")
print("-" * 60)

print("Saved model:", MODEL_PATH)

# Verify model can be loaded
loaded_model = joblib.load(MODEL_PATH)

verification_predictions = loaded_model.predict(
    X_test.head(5)
)

print("\nModel reload verification: PASSED")
print("Sample predictions:", verification_predictions.tolist())

# Save metadata for reproducibility
metadata = {
    "project": "Visit with Us - Tourism Package Prediction",
    "selected_model": best_model_name,
    "selection_metric": "5-fold stratified cross-validation F1-score",
    "best_cv_f1": float(best_overall_score),
    "best_parameters": best_parameters,
    "feature_columns": X_train.columns.tolist(),
    "target_column": TARGET_COLUMN,
    "random_state": RANDOM_STATE
}

metadata_path = os.path.join(
    DEPLOYMENT_DIR,
    "model_metadata.json"
)

with open(metadata_path, "w") as file:
    json.dump(metadata, file, indent=4, default=str)

# ============================================================
# 11. FINAL TRAINING SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("MODEL TRAINING AND REGISTRATION COMPLETED SUCCESSFULLY")
print("=" * 75)

print("Experiments completed:", len(experiment_results))
print("Best model:", best_model_name)
print("Best CV F1-score:", round(best_overall_score, 4))

print("\nGenerated deployment files:")
print("1.", MODEL_PATH)
print("2.", results_path)
print("3.", metadata_path)

print("\nMLflow experiment:", experiment_name)
print("MLflow tracking directory: ./mlruns")

print("\nThe trained model is ready for deployment.")
print("=" * 75)
