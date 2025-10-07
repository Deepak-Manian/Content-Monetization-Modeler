import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('linear_regression_model.pkl')

st.title("YouTube Ad Revenue Prediction")

# Numeric inputs
st.header("Numeric Features")
views = st.number_input("Views", min_value=0, value=1000)
likes = st.number_input("Likes", min_value=0, value=100)
comments = st.number_input("Comments", min_value=0, value=10)
watch_time_minutes = st.number_input("Watch Time (minutes)", min_value=0.0, value=500.0)
video_length_minutes = st.number_input("Video Length (minutes)", min_value=0.1, value=10.0)
subscribers = st.number_input("Subscribers", min_value=0, value=10000)

# Calculate engagement rate on the fly for user convenience
engagement_rate = (likes + comments) / views if views > 0 else 0

st.write(f"Engagement Rate (calculated): {engagement_rate:.4f}")

# Categorical inputs
st.header("Categorical Features")
category = st.selectbox("Category", ["Gaming", "Education", "Entertainment", "Music", "Sports", "News", "Other"])
device = st.selectbox("Device", ["Mobile", "Desktop", "Tablet"])
country = st.selectbox("Country", ["USA", "India", "UK", "Canada", "Australia", "Other"])

# Prepare input dict with all columns used in training
# IMPORTANT: Replace these with your actual encoded columns used in training, based on your one-hot encoding

all_columns = model.feature_names_in_

# Initialize all columns to 0 first
input_dict = {col: 0 for col in all_columns}

# Fill numeric features
input_dict.update({
    'views': views,
    'likes': likes,
    'comments': comments,
    'watch_time_minutes': watch_time_minutes,
    'video_length_minutes': video_length_minutes,
    'subscribers': subscribers,
    'engagement_rate': engagement_rate
})

# Encode categorical features
# Category columns — assuming your dummy columns start with "category_"
category_col = f"category_{category}"
if category_col in input_dict:
    input_dict[category_col] = 1

# Device columns
device_col = f"device_{device}"
if device_col in input_dict:
    input_dict[device_col] = 1

# Country columns
country_col = f"country_{country}"
if country_col in input_dict:
    input_dict[country_col] = 1

# Create dataframe in correct column order
input_df = pd.DataFrame([input_dict], columns=all_columns)

# Predict button
if st.button("Predict Ad Revenue"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Ad Revenue: ${prediction[0]:.2f}")
