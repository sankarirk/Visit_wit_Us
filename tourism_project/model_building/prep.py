
import os
import pandas as pd

from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
# Define paths and static hyperparameter settings for reproducibility
DATA_PATH = "tourism_project/data/tourism.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "ProdTaken"

# --------------------------------------------------
# 2. Load the registered dataset
# --------------------------------------------------
# Verify dataset existence before attempting loading procedures
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

# Read raw CSV data into a pandas DataFrame
df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("DATA PREPARATION")
print("=" * 65)
print("\nOriginal dataset shape:", df.shape)

# --------------------------------------------------
# 3. Validate the dataset
# --------------------------------------------------
# Check for target column presence to prevent downstream errors
if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' is missing."
    )

# Ensure there are no missing values in the target column
if df[TARGET_COLUMN].isnull().any():
    raise ValueError(
        "Target column contains missing values."
    )

# Confirm that the target has exactly two classes for binary classification
if df[TARGET_COLUMN].nunique() != 2:
    raise ValueError(
        "ProdTaken must contain exactly two target classes."
    )

# --------------------------------------------------
# 4. Remove unnecessary columns
# --------------------------------------------------
# Specify target index and metadata columns to drop from model inputs
columns_to_drop = [
    "Unnamed: 0",
    "CustomerID"
]

# Only drop columns that are actually present in the loaded dataset
existing_columns_to_drop = [
    col for col in columns_to_drop
    if col in df.columns
]

df = df.drop(columns=existing_columns_to_drop)

print("\nRemoved unnecessary columns:")
print(existing_columns_to_drop)
print("\nDataset shape after column removal:", df.shape)

# --------------------------------------------------
# 5. Separate features and target
# --------------------------------------------------
# Segregate independent feature space (X) from the dependent target vector (y)
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

print("\nNumber of input features:", X.shape[1])
print("Target column:", TARGET_COLUMN)
print("\nFeature columns:")
print(X.columns.tolist())

# --------------------------------------------------
# 6. Split into training and testing sets
# --------------------------------------------------
# Partition dataset using stratified splitting to preserve original class ratios
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# --------------------------------------------------
# 7. Save the four split files
# --------------------------------------------------
# Serialize training and testing splits to disk for use in subsequent pipeline jobs
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)

ytrain.to_frame(name=TARGET_COLUMN).to_csv(
    "ytrain.csv",
    index=False
)

ytest.to_frame(name=TARGET_COLUMN).to_csv(
    "ytest.csv",
    index=False
)

# --------------------------------------------------
# 8. Display preparation summary
# --------------------------------------------------
# Print summary logs detailing the final split shapes and distributions
print("\nTraining features shape:", Xtrain.shape)
print("Testing features shape:", Xtest.shape)
print("Training target shape:", ytrain.shape)
print("Testing target shape:", ytest.shape)

print("\nTraining target distribution:")
print(ytrain.value_counts(normalize=True).mul(100).round(2))

print("\nTesting target distribution:")
print(ytest.value_counts(normalize=True).mul(100).round(2))

print("\nSaved files:")
print("1. Xtrain.csv")
print("2. Xtest.csv")
print("3. ytrain.csv")
print("4. ytest.csv")

print("\n" + "=" * 65)
print("DATA PREPARATION COMPLETED SUCCESSFULLY")
print("=" * 65)
