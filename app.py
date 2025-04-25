import streamlit as st
import streamlit.components.v1 as components
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
    with st.expander("**Dataset Description**"):
        st.write("""
        Each row in the dataset represents a single house sale transaction, and each column contains a specific attribute related to the house's physical characteristics, condition, or sale details.

        The dataset includes information about:

        - Property characteristics such as size of living area, number of bedrooms above ground, basement size, and garage area
        - Condition and quality ratings such as overall material quality and kitchen quality
        - Construction features such as year built, year remodeled, and presence of porches or decks
        - Sale information such as the final selling price

        | Variable          | Meaning                                          | Units/Values                                 |
        |-------------------|--------------------------------------------------|----------------------------------------------|
        | 1stFlrSF          | First floor square footage                       | Square feet (334–4692)                       |
        | 2ndFlrSF          | Second floor square footage                      | Square feet (0–2065)                         |
        | BedroomAbvGr      | Bedrooms above ground (not including basement)   | Integer (0–8)                                |
        | BsmtExposure      | Exposure level of basement walls                 | Gd, Av, Mn, No, None                         |
        | BsmtFinSF1        | Finished basement area (Type 1)                  | Square feet (0–5644)                         |
        | BsmtUnfSF         | Unfinished basement area                         | Square feet (0–2336)                         |
        | TotalBsmtSF       | Total basement area                              | Square feet (0–6110)                         |
        | GarageArea        | Garage size                                      | Square feet (0–1418)                         |
        | GarageFinish      | Garage interior finish                           | Fin, RFn, Unf, None                          |
        | GarageYrBlt       | Year the garage was built                        | Year (1900–2010)                             |
        | GrLivArea         | Above-grade living area                          | Square feet (334–5642)                       |
        | KitchenQual       | Kitchen quality rating                           | Ex, Gd, TA, Fa, Po                           |
        | LotArea           | Lot size                                         | Square feet (1300–215245)                    |
        | LotFrontage       | Street frontage length                           | Feet (21–313)                                |
        | MasVnrArea        | Masonry veneer area                              | Square feet (0–1600)                         |
        | EnclosedPorch     | Enclosed porch area                              | Square feet (0–286)                          |
        | OpenPorchSF       | Open porch area                                  | Square feet (0–547)                          |
        | OverallCond       | Overall condition of the house                   | 1 (Very Poor) to 10 (Very Excellent)         |
        | OverallQual       | Overall material and finish quality              | 1 (Very Poor) to 10 (Very Excellent)         |
        | WoodDeckSF        | Wood deck area                                   | Square feet (0–736)                          |
        | YearBuilt         | Year the house was originally built              | Year (1872–2010)                             |
        | YearRemodAdd      | Year the house was last remodeled                | Year (1950–2010)                             |
        | SalePrice         | Final sale price of the property                 | USD (34900–755000)                           |

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
    st.markdown("""
                The most important features are highly connected with the size and quality of the house.
                The overall quality of the house is the most important feature, indicating that buyers in the area are valuing subjective elements when purchasing a home. 
                It is followed by the above-ground living area, garage area, total basement area, which makes sense since they form the bulk of the house's total area. 
                The size of a home is the main information displayed in advertisements alongside location, which can be seen here in this model. 
                Lastly we have the year the house was remodeled. It can affected the price of the house, as it is a good indicator of how well the house has been maintained and which technologies it has, such as insulation or air conditioning.
    """)
    st.markdown("These are the features that most influenced the model’s predictions:")
    st.write(pd.DataFrame(model_metrics["feature_importance"]))

    with st.expander("Pandas Profile Report"):
        st.write("The dataset was analyzed using Pandas Profiling, which provides insights into the data distribution, correlations, and missing values.")
        st.markdown("The report is available for download:")

        with open("outputs/reports/data_profile_report.pdf", "rb") as file:
            btn = st.download_button(
                label="📥 Download Pandas Profiling Report in PDF",
                data=file,
                file_name="data_profile_report.pdf",
                mime="application/pdf"
            )

        with open("outputs/reports/data_profile_report.html", "rb") as file:
            btn = st.download_button(
                label="📥 Download Pandas Profiling Report in HTML",
                data=file,
                file_name="data_profile_report.html",
                mime="text/html"
            )
        
        with open("outputs/reports/data_profile_report.html",'r') as f: 
            html_data = f.read()

            # Show in webpage
            st.header("Pandas Profiling Report")
            st.components.v1.html(html_data, height=800, scrolling=True)

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
    st.markdown("### House sales price from client's inherited houses")
    # Load the dataset of inherited houses
    inherited_houses_path = Path(__file__).parent / 'inputs' / 'datasets' / 'raw' / 'house-price-20211124T154130Z-001' / 'house-price' / 'inherited_houses.csv'
    inherited_houses = pd.read_csv(inherited_houses_path)

    # Predict prices for the inherited houses
    inherited_houses["PredictedPrice"] = pipeline.predict(inherited_houses[features])

    # Display the predictions
    st.markdown("### Predicted Prices for Inherited Houses:")
    st.dataframe(inherited_houses[["GrLivArea", "OverallQual", "GarageArea", "TotalBsmtSF", "YearRemodAdd", "PredictedPrice"]])

    # Calculate and display the total predicted price
    total_price = inherited_houses["PredictedPrice"].sum()
    st.markdown(f"The sum of predicted prices for the four houses is:🏷️ **US${round(total_price, 2):,}**")

    st.markdown("---")

    # Collect user input
    GarageArea = st.number_input("Garage Area (GarageArea) [sq ft]", min_value=0, value=400)
    GrLivArea = st.number_input("Above-Ground Living Area (GrLivArea) [sq ft]", min_value=0, value=1500)
    OverallQual = st.selectbox("Overall Quality (OverallQual) [1 = Very Poor, 10 = Excellent]", options=list(range(1, 11)), index=5)
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
    st.markdown("### How to Read the Scatterplot")
    st.info("""
            - The blue dot indicates the actual value and its predicted value provided by the ML Pipeline for a given datapoint. The red line indicates where the predicted value is the actual value.
            - Ideally, the blue dots should follow along the red line. We note this iverall trend in the plots velow, for both Train and Test sets. We note there are few datapoints when teh actual price is greater than 500,000, and for these datapoints, the model tends to under estimate the sale price. This is a limitation of the model, as it is not able to predict these values accurately., and 
    """, icon="ℹ️")
    st.image("outputs/ml_pipeline/predict_house_price/v2/regression_scatterplot.png", caption="Actual vs Predicted Scatterplot")
