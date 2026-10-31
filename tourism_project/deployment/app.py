
import json
import joblib
import pandas as pd
import streamlit as st

from pathlib import Path

# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Visit with Us - Package Prediction",
    page_icon="✈️",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

MODEL_PATH = APP_DIR / "best_model.joblib"
METADATA_PATH = APP_DIR / "model_metadata.json"
DATA_PATH = PROJECT_DIR / "data" / "tourism.csv"

# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Reference dataset not found: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Model metadata not found: {METADATA_PATH}"
        )

    with open(METADATA_PATH, "r") as file:
        return json.load(file)


# ============================================================
# 3. APPLICATION HEADER
# ============================================================

st.title("✈️ Visit with Us")
st.subheader("Wellness Tourism Package Purchase Prediction")

st.write(
    """
    Welcome to Visit with Us!

    This application uses a trained Machine Learning model
    to predict whether a customer is likely to purchase
    a wellness tourism package.

    Enter the customer details below and click
    **Predict Package Purchase** to view the result.
    """
)

st.divider()

# ============================================================
# 4. LOAD MODEL AND REFERENCE DATA
# ============================================================

try:
    model = load_model()
    reference_df = load_reference_data()
    metadata = load_metadata()

    feature_columns = metadata["feature_columns"]

except Exception as error:
    st.error(f"Application initialization failed: {error}")
    st.stop()


# ============================================================
# 5. DEFINE INPUT FEATURE TYPES
# ============================================================

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation"
]

numeric_features = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome"
]

# Check that the expected features are available
if set(feature_columns) != set(
    categorical_features + numeric_features
):
    st.error(
        "The application feature configuration does not "
        "match the trained model metadata."
    )
    st.stop()


# ============================================================
# 6. COLLECT CUSTOMER INPUTS
# ============================================================

st.header("Customer Information")

customer_input = {}

with st.form("customer_prediction_form"):

    st.subheader("Personal and Customer Details")

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # Numerical inputs
    # --------------------------------------------------------

    for index, feature in enumerate(numeric_features):

        column = col1 if index % 2 == 0 else col2

        series = pd.to_numeric(
            reference_df[feature],
            errors="coerce"
        ).dropna()

        if series.empty:
            raise ValueError(
                f"No valid reference values for {feature}"
            )

        minimum = float(series.min())
        maximum = float(series.max())
        median = float(series.median())

        is_integer = pd.api.types.is_integer_dtype(
            reference_df[feature]
        )

        with column:

            if is_integer:
                customer_input[feature] = st.number_input(
                    feature,
                    min_value=int(minimum),
                    max_value=int(maximum),
                    value=int(median),
                    step=1,
                    key=f"input_{feature}"
                )

            else:
                customer_input[feature] = st.number_input(
                    feature,
                    min_value=minimum,
                    max_value=maximum,
                    value=median,
                    step=0.1,
                    key=f"input_{feature}"
                )

    st.divider()

    st.subheader("Customer Preferences")

    col3, col4 = st.columns(2)

    # --------------------------------------------------------
    # Categorical inputs
    # --------------------------------------------------------

    for index, feature in enumerate(categorical_features):

        column = col3 if index % 2 == 0 else col4

        categories = (
            reference_df[feature]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        categories = sorted(categories)

        if not categories:
            raise ValueError(
                f"No categories available for {feature}"
            )

        with column:

            customer_input[feature] = st.selectbox(
                feature,
                options=categories,
                key=f"input_{feature}"
            )

    st.divider()

    submitted = st.form_submit_button(
        "Predict Package Purchase",
        use_container_width=True
    )


# ============================================================
# 7. GENERATE PREDICTION
# ============================================================

if submitted:

    try:

        # Create input DataFrame in the exact training order
        input_df = pd.DataFrame(
            [customer_input],
            columns=feature_columns
        )

        st.subheader("Customer Input Summary")

        st.dataframe(
            input_df,
            use_container_width=True
        )

        # Predict purchase outcome
        prediction = model.predict(input_df)[0]

        st.divider()

        st.header("Prediction Result")

        if int(prediction) == 1:

            st.success(
                "The customer is predicted to purchase "
                "the wellness tourism package."
            )

        else:

            st.info(
                "The customer is predicted not to purchase "
                "the wellness tourism package."
            )

        # Display purchase probability
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_df)[0]

            purchase_probability = float(probabilities[1])

            non_purchase_probability = float(probabilities[0])

            col5, col6 = st.columns(2)

            with col5:
                st.metric(
                    "Purchase Probability",
                    f"{purchase_probability * 100:.2f}%"
                )

            with col6:
                st.metric(
                    "Non-Purchase Probability",
                    f"{non_purchase_probability * 100:.2f}%"
                )

            st.progress(purchase_probability)

        st.caption(
            "Prediction generated using the trained "
            "Random Forest machine learning pipeline."
        )

    except Exception as error:

        st.error(
            f"An error occurred while generating the prediction: {error}"
        )


# ============================================================
# 8. APPLICATION FOOTER
# ============================================================

st.divider()

st.caption(
    "Visit with Us | Wellness Tourism Package Prediction "
    "| Machine Learning Deployment Project"
)
