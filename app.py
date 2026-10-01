
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score
)

# --------------------------------------------------
# 1. PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Real Estate Investment Analyzer",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Real Estate Investment Analyzer")
st.write(
    "Explore properties, estimate future prices, "
    "and analyze investment classifications."
)

DATA_PATH = Path(__file__).parent / "cleaned_housing_data.csv"

# --------------------------------------------------
# 2. LOAD DATA AND TRAIN MODELS
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    required = [
        "City", "Property_Type", "BHK",
        "Size_in_SqFt", "Price_in_Lakhs",
        "Price_per_SqFt", "Future_Price_5Y_Lakhs",
        "Good_Investment"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df


@st.cache_resource
def train_models(data):
    df = data.copy()

    # Avoid high-cardinality identifier/text columns.
    excluded = {
        "ID", "Locality", "Amenities",
        "Price_per_SqFt",
        "Future_Price_5Y_Lakhs",
        "Good_Investment"
    }

    # Use a sample to reduce training time.
    sample = df.sample(
        n=min(30000, len(df)),
        random_state=42
    ).copy()

    # Regression features
    regression_features = [
        c for c in sample.columns
        if c not in excluded
    ]

    # Classification features
    # Price_per_SqFt is excluded because it directly defines
    # the current Good_Investment target.
    classification_features = [
        c for c in sample.columns
        if c not in excluded
    ]

    # Prepare targets
    reg_data = sample.dropna(
        subset=["Future_Price_5Y_Lakhs"]
    ).copy()

    cls_data = sample.dropna(
        subset=["Good_Investment"]
    ).copy()

    # Prepare regression input
    X_reg = reg_data[regression_features].copy()
    y_reg = reg_data["Future_Price_5Y_Lakhs"]

    # Prepare classification input
    X_cls = cls_data[classification_features].copy()
    y_cls = cls_data["Good_Investment"].astype(int)

    # Clean missing values consistently
    for frame in [X_reg, X_cls]:
        for col in frame.columns:
            if (
                pd.api.types.is_object_dtype(frame[col])
                or isinstance(frame[col].dtype, pd.CategoricalDtype)
            ):
                frame[col] = frame[col].fillna("Unknown").astype(str)
            else:
                frame[col] = pd.to_numeric(
                    frame[col], errors="coerce"
                )
                median = frame[col].median()
                frame[col] = frame[col].fillna(
                    median if pd.notna(median) else 0
                )

    # Convert text categories into numeric columns
    X_reg = pd.get_dummies(X_reg, dummy_na=True)
    X_cls = pd.get_dummies(X_cls, dummy_na=True)

    # Store the training column names for future predictions
    reg_columns = X_reg.columns.tolist()
    cls_columns = X_cls.columns.tolist()

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42,
        stratify=y_cls
    )

    reg_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=18,
        random_state=42,
        n_jobs=-1
    )

    cls_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=18,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    )

    reg_model.fit(Xr_train, yr_train)
    cls_model.fit(Xc_train, yc_train)

    metrics = {
        "mae": mean_absolute_error(
            yr_test, reg_model.predict(Xr_test)
        ),
        "r2": r2_score(
            yr_test, reg_model.predict(Xr_test)
        ),
        "accuracy": accuracy_score(
            yc_test, cls_model.predict(Xc_test)
        )
    }

    return (
        reg_model, cls_model,
        regression_features, classification_features,
        reg_columns, cls_columns, metrics
    )


# --------------------------------------------------
# 3. LOAD AND TRAIN
# --------------------------------------------------

try:
    df = load_data()

    (
        reg_model, cls_model,
        reg_features, cls_features,
        reg_columns, cls_columns, metrics
    ) = train_models(df)

except Exception as error:
    st.error(f"Could not load data or train models: {error}")
    st.stop()


# --------------------------------------------------
# 4. DASHBOARD SUMMARY
# --------------------------------------------------

c1, c2, c3 = st.columns(3)

c1.metric("Total Properties", f"{len(df):,}")
c2.metric(
    "Average Property Price",
    f"₹{df['Price_in_Lakhs'].mean():.2f} Lakhs"
)
c3.metric(
    "Good Investment Labels",
    f"{(df['Good_Investment'].mean() * 100):.1f}%"
)

st.divider()


# --------------------------------------------------
# 5. PROPERTY FILTERS
# --------------------------------------------------

st.header("🔎 Find Properties")

cities = sorted(df["City"].dropna().astype(str).unique())
types = sorted(df["Property_Type"].dropna().astype(str).unique())

col1, col2, col3 = st.columns(3)

with col1:
    selected_city = st.selectbox("Select City", ["All"] + cities)

with col2:
    selected_type = st.selectbox(
        "Property Type", ["All"] + types
    )

with col3:
    bhk_values = sorted(
        pd.to_numeric(df["BHK"], errors="coerce").dropna().unique()
    )
    selected_bhk = st.selectbox(
        "BHK", ["All"] + [int(x) for x in bhk_values]
    )

min_area = int(max(0, df["Size_in_SqFt"].min()))
max_area = int(df["Size_in_SqFt"].max())
min_price = float(df["Price_in_Lakhs"].min())
max_price = float(df["Price_in_Lakhs"].max())

col1, col2 = st.columns(2)

with col1:
    area_range = st.slider(
        "Property Area (SqFt)",
        min_value=min_area,
        max_value=max_area,
        value=(min_area, max_area)
    )

with col2:
    price_range = st.slider(
        "Price Range (Lakhs)",
        min_value=float(np.floor(min_price)),
        max_value=float(np.ceil(max_price)),
        value=(
            float(np.floor(min_price)),
            float(np.ceil(max_price))
        )
    )

filtered = df[
    df["Size_in_SqFt"].between(*area_range)
    & df["Price_in_Lakhs"].between(*price_range)
].copy()

if selected_city != "All":
    filtered = filtered[
        filtered["City"].astype(str) == selected_city
    ]

if selected_type != "All":
    filtered = filtered[
        filtered["Property_Type"].astype(str) == selected_type
    ]

if selected_bhk != "All":
    filtered = filtered[
        pd.to_numeric(filtered["BHK"], errors="coerce")
        == selected_bhk
    ]

st.subheader(f"Matching Properties: {len(filtered):,}")

display_cols = [
    c for c in [
        "City", "Locality", "Property_Type", "BHK",
        "Size_in_SqFt", "Price_in_Lakhs",
        "Price_per_SqFt", "Good_Investment"
    ]
    if c in filtered.columns
]

st.dataframe(
    filtered[display_cols].head(500),
    use_container_width=True
)

st.caption("Showing at most 500 matching rows in the table.")


# --------------------------------------------------
# 6. PROPERTY PREDICTION FORM
# --------------------------------------------------

st.divider()
st.header("🔮 Predict a Property")

st.write("Enter the property details below.")

with st.form("property_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        city = st.selectbox("Prediction City", cities)
        property_type = st.selectbox("Prediction Property Type", types)
        bhk = st.number_input("BHK", 1, 20, 2)
        size = st.number_input("Area (SqFt)", 100, 20000, 1000)

    with col2:
        current_price = st.number_input(
            "Current Price (Lakhs)", 1.0, 100000.0, 50.0
        )
        year_built = st.number_input(
            "Year Built", 1900, 2026, 2018
        )
        floor_no = st.number_input("Floor Number", 0, 200, 2)
        total_floors = st.number_input(
            "Total Floors", 1, 200, 5
        )

    with col3:
        furnished_options = (
            sorted(df["Furnished_Status"].dropna().astype(str).unique())
            if "Furnished_Status" in df else ["Unknown"]
        )
        furnished = st.selectbox(
            "Furnished Status", furnished_options
        )

        security_options = (
            sorted(df["Security"].dropna().astype(str).unique())
            if "Security" in df else ["Unknown"]
        )
        security = st.selectbox("Security", security_options)

        parking_options = (
            sorted(df["Parking_Space"].dropna().astype(str).unique())
            if "Parking_Space" in df else ["Unknown"]
        )
        parking = st.selectbox("Parking", parking_options)

        transport_options = (
            sorted(
                df["Public_Transport_Accessibility"]
                .dropna().astype(str).unique()
            )
            if "Public_Transport_Accessibility" in df
            else ["Unknown"]
        )
        transport = st.selectbox(
            "Public Transport", transport_options
        )

    submitted = st.form_submit_button(
        "Analyze Property",
        type="primary"
    )


if submitted:
    input_data = {}

    # Fill every feature expected by the model.
    for col in set(reg_features + cls_features):
        if col in df.columns:
            if (
                pd.api.types.is_numeric_dtype(df[col])
                or pd.api.types.is_bool_dtype(df[col])
            ):
                median = pd.to_numeric(
                    df[col], errors="coerce"
                ).median()
                input_data[col] = (
                    median if pd.notna(median) else 0
                )
            else:
                input_data[col] = "Unknown"
        else:
            input_data[col] = "Unknown"

    supplied = {
        "City": city,
        "Property_Type": property_type,
        "BHK": bhk,
        "Size_in_SqFt": size,
        "Price_in_Lakhs": current_price,
        "Year_Built": year_built,
        "Floor_No": floor_no,
        "Total_Floors": total_floors,
        "Furnished_Status": furnished,
        "Security": security,
        "Parking_Space": parking,
        "Public_Transport_Accessibility": transport,
        "Age_of_Property": max(0, 2026 - year_built),
        "Price_per_SqFt": current_price * 100000 / size
    }

    input_data.update(supplied)

    # Predict five-year price
    reg_input = pd.DataFrame([{c: input_data.get(c, "Unknown")
                               for c in reg_features}])

    for col in reg_input.columns:
        if col in df.columns and (
            pd.api.types.is_numeric_dtype(df[col])
            or pd.api.types.is_bool_dtype(df[col])
        ):
            reg_input[col] = pd.to_numeric(
                reg_input[col], errors="coerce"
            ).fillna(0)
        else:
            reg_input[col] = reg_input[col].fillna("Unknown").astype(str)

    reg_input = pd.get_dummies(reg_input, dummy_na=True)
    reg_input = reg_input.reindex(
        columns=reg_columns, fill_value=0
    )

    predicted_price = float(reg_model.predict(reg_input)[0])

    # Predict investment classification
    cls_input = pd.DataFrame([{c: input_data.get(c, "Unknown")
                               for c in cls_features}])

    for col in cls_input.columns:
        if col in df.columns and (
            pd.api.types.is_numeric_dtype(df[col])
            or pd.api.types.is_bool_dtype(df[col])
        ):
            cls_input[col] = pd.to_numeric(
                cls_input[col], errors="coerce"
            ).fillna(0)
        else:
            cls_input[col] = cls_input[col].fillna("Unknown").astype(str)

    cls_input = pd.get_dummies(cls_input, dummy_na=True)
    cls_input = cls_input.reindex(
        columns=cls_columns, fill_value=0
    )

    prediction = int(cls_model.predict(cls_input)[0])
    probabilities = cls_model.predict_proba(cls_input)[0]
    class_index = list(cls_model.classes_).index(prediction)
    confidence = probabilities[class_index] * 100

    st.subheader("Prediction Results")

    a, b, c = st.columns(3)

    a.metric(
        "Estimated Price After 5 Years",
        f"₹{predicted_price:,.2f} Lakhs"
    )

    b.metric(
        "Investment Classification",
        "Good Investment" if prediction == 1
        else "Not Good Investment"
    )

    c.metric("Model Confidence", f"{confidence:.1f}%")

    st.caption(
        "Confidence is the model's predicted class probability, "
        "not a guarantee of investment success."
    )


# --------------------------------------------------
# 7. CITY-WISE PRICE CHART
# --------------------------------------------------

st.divider()
st.header("📊 Visual Insights")

city_prices = (
    df.groupby("City", as_index=False)["Price_in_Lakhs"]
    .mean()
    .sort_values("Price_in_Lakhs", ascending=False)
)

st.subheader("Average Property Price by City")

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(city_prices["City"], city_prices["Price_in_Lakhs"])
ax.set_xlabel("City")
ax.set_ylabel("Average Price (Lakhs)")
ax.tick_params(axis="x", rotation=90)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------
# 8. CORRELATION HEATMAP
# --------------------------------------------------

st.subheader("Property Feature Correlation Heatmap")

numeric_df = df.select_dtypes(include=np.number)

heatmap_cols = [
    c for c in [
        "BHK", "Size_in_SqFt", "Price_in_Lakhs",
        "Price_per_SqFt", "Age_of_Property",
        "Nearby_Schools", "Nearby_Hospitals",
        "Future_Price_5Y_Lakhs"
    ]
    if c in numeric_df.columns
]

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    numeric_df[heatmap_cols].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=ax
)
st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------
# 9. MODEL PERFORMANCE
# --------------------------------------------------

st.divider()
st.header("🤖 Model Performance")

m1, m2, m3 = st.columns(3)

m1.metric(
    "Regression MAE",
    f"₹{metrics['mae']:.2f} Lakhs"
)

m2.metric(
    "Regression R²",
    f"{metrics['r2']:.3f}"
)

m3.metric(
    "Classification Accuracy",
    f"{metrics['accuracy'] * 100:.2f}%"
)

st.caption(
    "Metrics are calculated on held-out rows from the available dataset. "
    "They measure agreement with the dataset's generated targets, "
    "not verified future market outcomes."
)


# --------------------------------------------------
# 10. FEATURE IMPORTANCE
# --------------------------------------------------

st.subheader("Regression: Feature Importance")

reg_importance = pd.DataFrame({
    "Feature": reg_columns,
    "Importance": reg_model.feature_importances_
}).sort_values("Importance", ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(
    reg_importance["Feature"][::-1],
    reg_importance["Importance"][::-1]
)
ax.set_xlabel("Importance")
ax.set_title("Top 15 Regression Features")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


st.subheader("Classification: Feature Importance")

cls_importance = pd.DataFrame({
    "Feature": cls_columns,
    "Importance": cls_model.feature_importances_
}).sort_values("Importance", ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(
    cls_importance["Feature"][::-1],
    cls_importance["Importance"][::-1]
)
ax.set_xlabel("Importance")
ax.set_title("Top 15 Classification Features")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


st.divider()
st.caption(
    "Educational project only. Future prices depend on market conditions, "
    "location, demand, interest rates and other factors not captured here."
)