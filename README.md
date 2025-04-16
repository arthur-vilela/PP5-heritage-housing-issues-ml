# Predictive Analytics Project – Heritage Housing Issues

Lydia Doe, a fictional individual, inherited four houses in Ames, Iowa. She needs help estimating their market value and understanding what features influence house prices in that region.

## 1. Dataset Description *(to be completed after data exploration)*

> *This section will describe the*
>
> - *content*
> - *structure*
> - *key characteristics of the dataset.*
>
> *It will include a summary of*
>
> - *variables*
> - *missing values*
> - *basic descriptive statistics.*

---

## 2. Business Requirements

| Requirement ID | Business Requirement |
|----------------|----------------------|
| **BR1**        | The client is interested in discovering how house attributes correlate with the sale price. The client expects data visualisations of the correlated variables against the sale price. |
| **BR2**        | The client is interested in predicting the house sale price from her four inherited houses, and any other house in Ames, Iowa. |

---

## 3. Hypotheses *(to be defined during data understanding)*

---

## 4. Mapping Business Requirements to ML Tasks and Dashboard

| Business Requirement | ML / Data Task                                               | Dashboard Feature                                 |
|----------------------|--------------------------------------------------------------|--------------------------------------------------|
| **BR1**              | Perform correlation analysis and feature exploration. Visualise top features that impact SalePrice. | “Data Insights” page with plots and explanations |
| **BR2**              | Train and evaluate a regression model. Provide predictions based on user input.                     | “Predict Price” page with input form and prediction output |

---

## 5. Business Case for the Machine Learning Task

Lydia Doe needs a reliable way to estimate housing prices in a region she is unfamiliar with. By training a machine learning model using historical housing data from Ames, Iowa, we can automate the valuation process and offer data-driven insights.

This model will be used in a public-facing dashboard where users (like Lydia) can:

- View how various attributes impact sale prices
- Input custom values to get instant price predictions

The goal is to build a solution that is both **accurate** and **interpretable**, making it accessible and valuable for non-technical users.

---

## 6. Model Objective, Metrics & Evaluation Strategy

---

## 7. Dashboard Design *(to be developed as app pages are implemented)*

> *This section will outline the planned layout of the Streamlit dashboard, including navigation structure and page content.*

**Planned Pages:**

- **Data Insights**  
  *Visuals and explanations of key features that affect sale prices*

- **Predict Price**  
  *Interactive form where users can enter property details and receive predictions*

- **Project Summary**  
  *Overview of objectives, methods, and findings*
