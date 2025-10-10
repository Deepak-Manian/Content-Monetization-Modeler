# Content Monetization Modeler 🎥💰

A machine learning project that predicts YouTube ad revenue based on video performance metrics and contextual features. This project implements multiple regression models and provides an interactive Streamlit web application for revenue predictions.

## 🎯 Project Overview

**Domain:** Social Media Analytics  
**Problem Statement:** Build a regression model to accurately estimate YouTube ad revenue for individual videos based on performance and contextual features, helping content creators and media companies with revenue forecasting and content strategy optimization.

## 🏆 Key Results

- **Best Model:** Linear Regression with **R² = 0.9526** and **RMSE = 13.47**
- **Dataset Size:** ~122,000 rows of YouTube video performance data
- **Model Accuracy:** All top 3 models achieved >95% R² score
- **Interactive App:** Fully functional Streamlit application for real-time predictions

## 🎯 Business Use Cases

- **Content Strategy Optimization:** Help creators determine content types with highest returns
- **Revenue Forecasting:** Enable media companies to predict income from future uploads
- **Creator Support Tools:** Integration into analytics platforms for YouTubers
- **Ad Campaign Planning:** Forecast ROI based on content performance metrics

## 🛠️ Technologies Used

- **Python 3.x**
- **Machine Learning:** Scikit-learn
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Web App:** Streamlit
- **Model Persistence:** Joblib, Pickle

## 📊 Dataset Information

- **Name:** YouTube Monetization Modeler Dataset
- **Format:** CSV
- **Size:** ~122,000 rows
- **Source:** Synthetic dataset created for learning purposes
- **Target Variable:** `ad_revenue_usd`

### Features:
- `video_id`: Unique identifier
- `date`: Upload/report date
- `views`, `likes`, `comments`: Performance metrics
- `watch_time_minutes`, `video_length_minutes`: Engagement metrics
- `subscribers`: Channel subscriber count
- `category`, `device`, `country`: Contextual information
- `ad_revenue_usd`: Revenue generated (target variable)

## 🚀 Installation & Setup

1. **Install required packages:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run app_yt.py
   ```

## 📁 Project Structure

```
content-monetization-modeler/
│
├── README.md                          # Project documentation
├── youtube_ad_revenue_dataset.csv     # Dataset
├── content_yt.ipynb                   # Main Jupyter notebook
├── app_yt.py                          # Streamlit web application
├── linear_regression_model.pkl        # Trained model file
└── requirements.txt                   # Python dependencies
```

## 🔬 Methodology

### 1. Data Preprocessing
- Handled ~5% missing values in key columns
- Removed ~2% duplicate records
- Encoded categorical variables (category, device, country)
- Created new feature: `engagement_rate = (likes + comments) / views`

### 2. Exploratory Data Analysis
- Comprehensive statistical analysis
- Correlation analysis between features
- Distribution analysis of target variable
- Outlier detection and handling

### 3. Model Building & Evaluation
Tested 5 different regression models:

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| **Lasso Regression** | **0.9526** | **13.47** | **3.12** |
| Ridge Regression | 0.9526 | 13.48 | 3.12 |
| Linear Regression | 0.9526 | 13.48 | 3.12 |
| Random Forest | 0.9521 | 13.55 | 3.70 |
| Gradient Boosting | 0.9518 | 13.58 | 4.07 |

### 4. Model Selection
**Lasso Regression** was selected as the best model based on:
- Highest R² score (0.9526)
- Lowest RMSE (13.47)
- Built-in feature selection capability
- Good generalization performance

## 💻 Streamlit App Features

The interactive web application includes:

- **Input Fields:** 
  - Numeric inputs (views, likes, comments, watch time, etc.)
  - Categorical selectors (category, device, country)
  - Auto-calculated engagement rate

- **Real-time Predictions:** 
  - Instant ad revenue predictions based on user inputs
  - Professional UI with clear result display

- **User-Friendly Interface:**
  - Intuitive input controls
  - Clear labeling and validation
  - Responsive design

## 📈 Key Insights

1. **Strong Predictive Power:** All models achieved >95% R² score, indicating excellent predictive capability
2. **Feature Importance:** Views, engagement metrics, and subscriber count are primary revenue drivers
3. **Model Consistency:** Linear models (Linear, Ridge, Lasso) performed similarly, suggesting linear relationships
4. **Engagement Rate:** The engineered feature significantly improved model interpretability

## 🎯 Skills Demonstrated

- **Machine Learning:** Regression modeling, model comparison, hyperparameter tuning
- **Data Science:** EDA, feature engineering, data cleaning, statistical analysis
- **Programming:** Python, Pandas, Scikit-learn, data visualization
- **Web Development:** Streamlit app development
- **Model Deployment:** Model persistence and loading for production use


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**  
Data Science Student at GUVI  
- LinkedIn: [LinkedIn](https://www.linkedin.com/in/deepak-manian/)

---

**⭐ If you found this project helpful, please give it a star!**
