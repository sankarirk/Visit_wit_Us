
import os
import pandas as pd

# Dataset location inside the project
DATA_PATH = "tourism_project/data/tourism.csv"

# Expected columns from the tourism dataset
EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome"
]

# Check whether the dataset exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}"
    )

# Load dataset
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DATA REGISTRATION REPORT")
print("=" * 60)

# Validate required columns
missing_columns = [
    column for column in EXPECTED_COLUMNS
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\nColumn validation: PASSED")
print("All expected columns are available.")

# Validate target column
if "ProdTaken" not in df.columns:
    raise ValueError("Target column ProdTaken is missing.")

# Dataset summary
print("\nDataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nTotal missing values:", df.isnull().sum().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nTarget distribution:")
print(df["ProdTaken"].value_counts(dropna=False))

print("\nTarget distribution (percentage):")
print(
    (df["ProdTaken"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\n" + "=" * 60)
print("DATA REGISTRATION COMPLETED SUCCESSFULLY")
print("=" * 60)
