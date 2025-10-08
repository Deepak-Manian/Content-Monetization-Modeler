import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Streamlit Page Config ----------
st.set_page_config(
    page_title="Content Monetization Modeler",
    layout="wide",
    page_icon="💰"
)

# ---------- Custom Dark Theme Styling ----------
st.markdown("""
    <style>
    body {background-color:#0e1117; color:#fff;}
    .sidebar .sidebar-content {background-color:#1c1c24;}
    h1, h2, h3, h4 { color: #F9FAFB !important; }
    .stNumberInput label, .stSelectbox label {
        color: #E0E0E0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Load Data ----------
DATA_PATH = 'I:/Project/youtube_ad_revenue_dataset.csv'  # Change if needed
df = pd.read_csv(DATA_PATH)

# Select numeric columns & handle missing values
numeric_df = df.select_dtypes(include=np.number)
numeric_df = numeric_df.fillna(numeric_df.mean())

# ---------- UI ----------
st.title("🔮 Predict YouTube Ad Revenue")

target_col = 'ad_revenue_usd'

if target_col not in numeric_df.columns:
    st.warning(f"❌ Target column '{target_col}' not found in dataset.")
else:
    st.subheader("📥 Enter Video Feature Values")

    feature_cols = numeric_df.columns.drop(target_col)
    input_data = {}

    # Optional: Create columns layout for cleaner UI
    cols_per_row = 3
    rows = (len(feature_cols) + cols_per_row - 1) // cols_per_row

    for i in range(rows):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i * cols_per_row + j
            if idx < len(feature_cols):
                col_name = feature_cols[idx]
                mean_val = float(numeric_df[col_name].mean())
                input_data[col_name] = cols[j].number_input(f"{col_name}", value=mean_val)

    # ---------- Prediction ----------
    if st.button("🚀 Predict"):
        X = numeric_df[feature_cols]
        y = numeric_df[target_col]

        model = LinearRegression()
        model.fit(X, y)

        input_df = pd.DataFrame([input_data])
        pred = model.predict(input_df)[0]
        pred = max(0, pred)  # Avoid negative revenue

        st.success(f"💰 **Predicted Ad Revenue:** ${pred:,.2f}")

        if pred == 0:
            st.info("This video may not generate ad revenue based on the current input values.")
