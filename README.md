# Predictive Analytics Project – Heritage Housing Issues

Lydia Doe, a fictional individual, inherited four houses in Ames, Iowa. She needs help estimating their market value and understanding what features influence house prices in that region.

## 1. Dataset Description *(to be completed after data exploration)*

## Dataset Content

The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data) and contains historical housing data for properties in Ames, Iowa. We created a fictitious user story where predictive analytics can be applied in a real project scenario.

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

This dataset provides sufficient structure and detail to build a machine learning model capable of predicting sale prices and to visualize how different attributes impact housing value in the Ames market.

---

## 2. Business Requirements

| Requirement ID | Business Requirement |
|----------------|----------------------|
| **BR1**        | The client is interested in discovering how house attributes correlate with the sale price. The client expects data visualisations of the correlated variables against the sale price. |
| **BR2**        | The client is interested in predicting the house sale price from her four inherited houses, and any other house in Ames, Iowa. |

---

## 3. Hypotheses

The following hypotheses were formed at the beginning of the project based on common assumptions in real estate and prior knowledge of the Ames housing dataset. These hypotheses will guide our exploratory analysis and help identify which features are most relevant for predicting house prices.

| # | Hypothesis                                                | Validation Method                                           |
|---|-----------------------------------------------------------|-------------------------------------------------------------|
| 1 | Houses with higher overall quality tend to sell for more. | Correlation and scatter plot (`OverallQual` vs `SalePrice`) |
| 2 | Larger above-ground living area leads to higher prices.   | Correlation and scatter plot (`GrLivArea` vs `SalePrice`)   |
| 3 | Renovated homes command higher prices.                    | Group comparison: `YearBuilt` ≠ `YearRemodAdd`              |
| 4 | Finished basements increase house value.                  | Group analysis: `BsmtFinType1` vs `SalePrice`               |
| 5 | Garage size positively affects sale price.                | Correlation and scatter plot (`GarageArea` vs `SalePrice`)  |

---

### The Rationale to Map the Business Requirements to the Data Visualizations and ML Tasks

- **Business Requirement 1: Data Visualization and Correlation Study**  
  - We will analyze the relationship between key house attributes and the final sale price.  
  - We will use correlation techniques (e.g., Pearson correlation) to identify the variables most strongly related to `SalePrice`.  
  - We will visualize the relationship between `SalePrice` and high-impact features such as `GrLivArea`, `OverallQual`, `GarageArea`, and others using scatter plots, box plots, and a correlation heatmap.  
  - We will use these insights to help the client (Lydia) better understand what features influence housing value and to support later feature selection for modeling.

- **Business Requirement 2: Regression Modeling and Prediction Interface**  
  - We will train a supervised regression model using selected numerical and categorical house attributes to predict the final sale price of a property.  
  - The model will be integrated into the Streamlit app, allowing the client to input property characteristics and receive a predicted sale price.  
  - We will evaluate the model's performance using regression metrics (R², MAE, RMSE) and ensure the output is interpretable and practically useful.  
  - This prediction tool will help Lydia estimate the value of her inherited properties and potential future acquisitions in Ames, Iowa.

---

## 4. Mapping Business Requirements to ML Tasks and Dashboard

| Business Requirement | ML / Data Task              | Dashboard Feature        |
|----------------------|-----------------------------|--------------------------|
| **BR1**              | Perform correlation analysis and feature exploration. Visualise top features that impact SalePrice. | “Data Insights” page with plots and explanations |
| **BR2**              | Train and evaluate a regression model. Provide predictions based on user input.                     | “Predict Price” page with input form and prediction output |

---

## 5. Business Case for the Machine Learning Task

Lydia Doe needs a reliable way to estimate housing prices in a region she is unfamiliar with. By training a machine learning model using historical housing data from Ames, Iowa, we can automate the valuation process and offer data-driven insights.

This model will be used in a public-facing dashboard where users (like Lydia) can:

- View how various attributes impact sale prices
- Input custom values to get instant price predictions

The goal is to build a solution that is both **accurate** and **interpretable**, making it accessible and valuable for non-technical users.

### Machine Learning Task Components

- **Aim:** Predict housing sale prices in Ames, Iowa, based on property attributes.
- **Learning Method:** Supervised regression using algorithms like Linear Regression or Random Forest.
- **Ideal Outcome:** A model that achieves strong predictive performance (R² ≥ 0.85) and is interpretable enough to provide trust in its estimates.
- **Success Metrics:**
  - R² Score (target: ≥ 0.85)
  - MAE and RMSE for error analysis
  - Visual Actual vs Predicted plots
- **Model Output:** SalePrice estimate for a house based on its features.
- **User Relevance:** Lydia can input house characteristics into the dashboard and receive an estimated market value.
- **Heuristics and Training Data:** The model was trained on a curated dataset from Ames, Iowa, publicly available on Kaggle, after undergoing thorough cleaning and feature engineering.

---

## 6. Model Objective, Metrics & Evaluation Strategy

---

## 7. Dashboard Design *(to be developed as app pages are implemented)*

The dashboard is designed to serve both non-technical stakeholders and technical users, and to clearly answer the business requirements. It is structured into four pages:

1. **Project Summary**  
   - Text explanation of the client's story and project context and goals.
   - Purpose: Inform users of the problem and solution scope.

2. **Housing Data Insights**  *(Answers BR1)*
   Answers Business Requirement 1 by showing how different property attributes correlate with sale price using visualizations and summary statistics.
   - Correlation heatmaps, scatter plots, boxplots.
   - Summary of top features influencing SalePrice.
   - Plot interpretations included.

3. **Predict House Price**  *(Answers BR2)*
   Allows users to input property characteristics and receive an estimated sale price using the trained machine learning model.
   - Interactive form for inputting house attributes.
   - Displays model-predicted sale price.

4. **Model Performance**  
   Provides technical details and evaluation metrics for the regression model, such as R² score and error distributions, supporting transparency and reproducibility.
   - Actual vs Predicted plots (Train/Test)
   - R², MAE, RMSE metrics
   - Summary stating if the model met the performance criteria.
