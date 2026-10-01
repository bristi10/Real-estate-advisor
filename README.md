# Real-estate-advisor
Real Estate Price Prediction and Investment Analysis is a Python-based project that analyzes housing data across Indian cities. It uses Pandas, Scikit-learn, and Streamlit to explore property prices, predict future housing prices, classify investment opportunities, and visualize market trends through an interactive dashboard


# 🏡 Real Estate Price Prediction and Investment Analysis

## 📌 Project Overview
This project analyzes housing prices across Indian cities using Python, Machine Learning, and Streamlit. It provides an interactive dashboard to explore property prices, predict future housing prices, and identify potential investment opportunities.

## 🎯 Objectives
- Analyze housing prices across different cities and property types.
- Explore relationships between property size, price, and investment potential.
- Predict future property prices using Machine Learning.
- Classify properties based on investment criteria.
- Visualize housing market trends through an interactive dashboard.

## 🛠️ Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Jupyter Notebook

## 📂 Project Structure
```text
real estate/
├── india_housing_prices.csv
├── cleaned_housing_data.csv
├── feature_engineering.py
├── eda.py
├── estate.py
└── README.md
```

## ⚙️ Key Features
- **Data Cleaning:** Handles missing values, duplicate records, and data types.
- **Feature Engineering:** Creates additional features such as property age, price per square foot, and amenities count.
- **Exploratory Data Analysis:** Examines property prices and housing trends.
- **Price Prediction:** Estimates property prices after five years using the project's defined growth assumptions and prediction model.
- **Investment Classification:** Classifies properties using the project's investment criteria.
- **Interactive Dashboard:** Provides filters, charts, predictions, and model evaluation metrics.

## 🚀 Installation and Setup

### 1. Clone the Repository
```bash
git clone <your-github-repository-url>
cd "real estate"
```

### 2. Install Dependencies
```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

### 3. Run the Application
```bash
python -m streamlit run estate.py
```

If your application file is named `app.py`, replace `estate.py` with `app.py`.

## 📊 Machine Learning
The project uses regression to estimate future property prices and classification to categorize properties according to the defined investment criteria. Model performance is evaluated using suitable regression and classification metrics.

## 📈 Expected Outcomes
- Better understanding of housing price patterns.
- Interactive exploration of property data.
- Estimated future property prices.
- Data-driven comparison of potential investment opportunities.

## ⚠️ Disclaimer
Future price estimates depend on the assumptions and models used in this project. Investment classifications are based on the project's defined criteria and should not be treated as guaranteed financial advice.

## 👩‍💻 Author
**Bhargabi Manna**

GitHub: [bristi10](https://github.com/bristi10)
