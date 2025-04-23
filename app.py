import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path

# === Load artifacts ===
version = "v1"
path = Path(__file__).parent / 'outputs' / 'ml_pipeline' / 'predict_house_price' / version
path_reports = Path(__file__).parent / 'outputs' / 'reports'

# Load pipeline and related artifacts
pipeline = joblib.load(path / 'pipeline_top5.pkl')
features = joblib.load(path / 'feature_list.pkl')

with open(path / 'model_metrics.json', "r") as f:
    model_metrics = json.load(f)

# === App Title and Navigation ===
st.set_page_config(page_title="House Price Prediction Dashboard", layout="wide")
st.title("🏡 Heritage Housing Issues – Dashboard")

page = st.sidebar.radio("Go to", ["Project Summary", "Housing Insights", "Predict Price", "Model Performance"])

# === 1. Summary Page ===
if page == "Project Summary":
    st.header("Project Summary")
    st.markdown("""
    Lydia Doe inherited four houses in **Ames, Iowa** and needs help estimating their market value.  
    This project applies **predictive analytics** to assist her in understanding **what features influence house prices** and provide reliable predictions.

    ### Business Requirements:
    - **BR1**: Discover how house attributes correlate with the sale price  
    - **BR2**: Predict the sale price of her four inherited houses (and others)

    ### Solution Overview:
    - Used a historical housing dataset from [Ames, Iowa](https://www.kaggle.com/codeinstitute/housing-prices-data)
    - Built several regression models and selected the best one using **GridSearchCV**
    - Final model: `ExtraTreesRegressor` (R² score ≈ 0.844 on test set)

    ### Dataset Details:
    - ~1,460 housing records with 20+ features
    - Features include:
        - Square footage (e.g., `GrLivArea`, `TotalBsmtSF`)
        - Quality and condition (e.g., `OverallQual`, `KitchenQual`)
        - Basement and garage features
        - Sale price as the target

    Navigate through the sidebar to explore the **top predictors**, try the **price prediction tool**, and view **model performance**.
    """)

# === 2. Housing Insights Page ===
elif page == "Housing Insights":
    st.header("Feature Importance")
    st.subheader("🔍 Top Predictors of House Price")
    st.markdown("""
    This page answers **Business Requirement 1 (BR1)**:  
    > _"Discover how house attributes correlate with the sale price."_

    The model identifies the most influential features in predicting a house's sale price in Ames, Iowa.  
    These are shown based on the **feature importance** values from the trained regression model.

    Higher importance means the model relied more on that feature when making predictions.
    """)
    st.image(f"{path}/feature_importance_top5.png", caption="Top 5 Most Important Features")

    st.markdown("These are the features that most influenced the model’s predictions:")
    st.write(pd.DataFrame(model_metrics["feature_importance"]))

    with st.expander("Pandas Profile Report"):
        st.write("The dataset was analyzed using Pandas Profiling, which provides insights into the data distribution, correlations, and missing values.")
        # st.image(f"{path}/pandas_profile_report.png", caption="Pandas Profile Report")
        st.markdown("The report is available for download:")
        st.markdown("[View Train Set Report](outputs/reports/data_profile_report.html)", unsafe_allow_html=True)

    with st.expander("Key Features"):
        st.write("""
                    - **Highly Correlated Features**:
                        - `GrLivArea` (Above-ground living area): Strong positive correlation with `SalePrice`.
                        - `GarageArea` (Garage size): Positively correlated, indicating larger garages increase house value.
                        - `TotalBsmtSF` (Total basement area): Larger basements are associated with higher prices.
                        - `1stFlrSF` (First-floor square footage): Positively correlated with `SalePrice`.
                        - `OverallQual` (Overall quality): Strong ordinal feature that significantly impacts `SalePrice`.
                    - **Temporal Features**:
                        - `YearBuilt` and `YearRemodAdd`: Newer or recently remodeled homes tend to have higher sale prices.
                    - **LotFrontage**:
                        - Indicates the linear feet of street connected to the property. Imputation and analysis suggest it has a moderate impact on `SalePrice`.
                """)	
    with st.expander("Feature Engineering"):
        st.write("""
                    - **Imputation Strategies**:
                        - Missing values in features like `GarageYrBlt` and `LotFrontage` were imputed using domain-specific strategies (e.g., median or constant values like `0` or `"None"`).
                        - Binary flags were added for missing values (e.g., `GarageYrBlt_missing`), which could provide predictive insights.
                    - **Transformations Applied**:
                        - Skewed numerical features were transformed to improve normality and reduce outliers:
                            - `GrLivArea`: Log transformation (`log_e`) reduced right-skew.
                            - `GarageArea`: Yeo-Johnson transformation improved symmetry.
                            - `TotalBsmtSF`: Power transformation improved distribution shape.
                            - `LotFrontage`: Yeo-Johnson transformation smoothed outliers.
                    - Ordinal encoding was applied to categorical features like `BsmtExposure`, `GarageFinish`, and `KitchenQual`.
                 """)
    with st.expander("Multicollinearity and Feature Selection"):
        st.write("""
                    - **SmartCorrelatedSelection**:
                    - The `SmartCorrelatedSelection` technique was used to identify and remove features with high multicollinearity, which can lead to overfitting and reduced model interpretability.
                        - "SmartCorrelatedSelection" was used with a correlation threshold = 0.8. For more details, refer to [SmartCorrelatedSelection Documentation](https://feature-engine.readthedocs.io/en/latest/selection/SmartCorrelatedSelection.html). Examples include:
                            - `GarageArea` (correlated with `TotalBsmtSF`).
                            - `1stFlrSF` (correlated with `GrLivArea`).
                            - `YearBuilt` and `YearRemodAdd` (correlated with each other).
                    - **Final Selected Features**:
                        - The notebook emphasizes retaining features with high predictive power while reducing redundancy. For example:
                            - ``OverallQual``, ``GrLivArea``, and ``TotalBsmtSF`` were retained due to their strong correlation with `SalePrice`.
                 """)
    with st.expander("Missing Data Patterns"):
        st.write("""
                    4. Missing Data Patterns
                    - **Garage Features**:
                        - Missing values in GarageYrBlt and GarageArea were associated with houses that lack a garage.
                    - **LotFrontage**:
                        - Missing values were imputed using the median grouped by Neighborhood, reflecting domain knowledge.
                 """)
    with st.expander("SalePrice Distribution"):
        st.write("""
                    - The target variable `SalePrice` likely exhibits right-skewness, as transformations like log_e and power were considered for features highly correlated with it. This suggests that higher-priced homes are less frequent in the dataset.
                """)

# === 3. Predict Price Page ===
elif page == "Predict Price":
    st.header("Predict House Sale Price")
    st.subheader("📦 Estimate the Sale Price of a Property")

    st.markdown("""
    This tool answers **Business Requirement 2 (BR2)**:  
    > _"Predict the sale price of the client’s inherited houses."_

    The model used here is based on an **ExtraTreesRegressor**, trained using the top 5 most predictive features.
    """)

    st.markdown("---")

    # Collect user input
    GrLivArea = st.number_input("Above-Ground Living Area (GrLivArea) [sq ft]", min_value=0, value=1500)
    OverallQual = st.selectbox("Overall Quality (OverallQual) [1 = Very Poor, 10 = Excellent]", options=list(range(1, 11)), index=5)
    GarageArea = st.number_input("Garage Area (GarageArea) [sq ft]", min_value=0, value=400)
    TotalBsmtSF = st.number_input("Total Basement Area (TotalBsmtSF) [sq ft]", min_value=0, value=800)
    YearRemodAdd = st.number_input("Remodel Year (YearRemodAdd)", min_value=1950, max_value=2025, value=2005)

    # Predict button
    if st.button("Predict"):
        input_df = pd.DataFrame([{
            "GrLivArea": GrLivArea,
            "OverallQual": OverallQual,
            "GarageArea": GarageArea,
            "TotalBsmtSF": TotalBsmtSF,
            "YearRemodAdd": YearRemodAdd
        }])

        # Make prediction
        prediction = pipeline.predict(input_df)[0]

        st.markdown("### Estimated Sale Price:")
        st.success(f"🏷️ **US${round(prediction, 2):,}**")

        # Download prediction as CSV
        input_df["PredictedPrice"] = prediction
        csv = input_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Prediction as CSV",
            data=csv,
            file_name="house_price_prediction.csv",
            mime="text/csv"
        )


# === 4. Model Performance Page ===
elif page == "Model Performance":
    st.header("Model Evaluation Metrics")
    st.markdown("""
    This page supports **Business Requirement 2 (BR2)**:  
    > _"Predict the sale price of the client’s inherited houses."_

    The model used is `ExtraTreesRegressor`, trained using GridSearchCV and selected for its high performance.

    ### Evaluation Metrics:
    - **R² (R-squared)**: How much variance in SalePrice is explained by the model
    - **MAE**: Mean Absolute Error – average prediction error
    - **RMSE**: Root Mean Squared Error – penalizes larger errors more heavily

    These scores are reported on both the **training** and **test** datasets to assess generalization.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Train Set")
        st.write(model_metrics["train"])

    with col2:
        st.subheader("Test Set")
        st.write(model_metrics["test"])

    st.markdown("---")
    st.subheader("Actual vs Predicted Plots")
    st.image("outputs/ml_pipeline/predict_house_price/v1/regression_scatterplot.png", caption="Actual vs Predicted Scatterplot")
