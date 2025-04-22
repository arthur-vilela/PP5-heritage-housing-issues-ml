import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt

# === Load artifacts ===
version = "v1"
path = f"C:/Users/Arthur/OneDrive/Documentos/Code Institute/PP5/PP5-heritage-housing-issues-ml/outputs/ml_pipeline/predict_house_price/{version}"

pipeline = joblib.load(f"{path}/pipeline_top5.pkl")
# pipeline = joblib.load(f"/pipeline_top5.pkl")
features = joblib.load(f"{path}/feature_list.pkl")

with open(f"{path}/model_metrics.json", "r") as f:
    model_metrics = json.load(f)

# === App Title and Navigation ===
st.set_page_config(page_title="House Price Prediction Dashboard", layout="wide")
st.title("🏡 Heritage Housing Issues – Dashboard")

page = st.sidebar.radio("Go to", ["Project Summary", "Housing Insights", "Predict Price", "Model Performance"])

# === 1. Summary Page ===
if page == "Project Summary":
    st.header("Project Summary")
    st.markdown("""
    Lydia Doe inherited four houses in Ames, Iowa. She needs help estimating their market value and understanding 
    which features influence sale price the most.

    This dashboard is designed to:
    - Visualize which housing attributes most impact price (BR1)
    - Allow users to input property info and get an instant price prediction (BR2)

    Built with a machine learning model trained on historical housing data.
    """)

# === 2. Housing Insights Page ===
elif page == "Housing Insights":
    st.header("Feature Importance")
    st.image(f"{path}/feature_importance_top5.png", caption="Top 5 Most Important Features")

    st.markdown("These are the features that most influenced the model’s predictions:")
    st.write(pd.DataFrame(model_metrics["feature_importance"]))

# === 3. Predict Price Page ===
elif page == "Predict Price":
    st.header("Predict House Sale Price")
    st.markdown("Fill in the property details below:")

    # Feature mapping for categorical variables (used in encoding step)
    mappings = {
        'KitchenQual': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
        'GarageFinish': {'Fin': 2, 'RFn': 1, 'Unf': 0, 'None': -1},
        'BsmtExposure': {'Gd': 3, 'Av': 2, 'Mn': 1, 'No': 0, 'None': -1},
        'BsmtFinType1': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': -1}
    }

    # Create dictionary to hold input values
    input_data = {}

    for feature in features:
        if feature in mappings:
            # Show dropdown with labels
            options = list(mappings[feature].keys())
            selected_label = st.selectbox(f"{feature}", options)
            input_data[feature] = mappings[feature][selected_label]
        else:
            # Show numeric input
            input_data[feature] = st.number_input(label=f"{feature}", step=1.0)

    # Predict button
    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])

        if input_df.isnull().any().any():
            st.error("Please fill in all fields to make a prediction.")
        else:
            prediction = pipeline.predict(input_df)[0]
            st.success(f"🏷️ Estimated Sale Price: ${round(prediction, 2):,}")


# === 4. Model Performance Page ===
elif page == "Model Performance":
    st.header("Model Evaluation Metrics")

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
