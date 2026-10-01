
from pathlib import Path
import pandas as pd

# Project settings
DATA_PATH = Path(__file__).parent / "india_housing_prices.csv"
OUTPUT_PATH = Path(__file__).parent / "cleaned_housing_data.csv"

CURRENT_YEAR = 2026
ANNUAL_GROWTH_RATE = 0.08
FORECAST_YEARS = 5


def load_and_preprocess_data(path=DATA_PATH):

    # 1. Load dataset
    df = pd.read_csv(path)
    print("Original dataset shape:", df.shape)

    # 2. Remove duplicate rows
    df = df.drop_duplicates().copy()

    # 3. Convert numeric columns into numbers
    numeric_cols = [
        "BHK",
        "Size_in_SqFt",
        "Price_in_Lakhs",
        "Year_Built",
        "Floor_No",
        "Total_Floors",
        "Nearby_Schools",
        "Nearby_Hospitals"
    ]

    # Convert columns to numeric
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Handle missing numeric values
    for col in numeric_cols:
        if col in df.columns:
            median_value = df[col].median()

            if pd.notna(median_value):
                df[col] = df[col].fillna(median_value)

    # 5. Handle missing categorical values
    categorical_cols = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    # 6. Calculate property age
    if "Year_Built" in df.columns:
        df["Age_of_Property"] = (
            CURRENT_YEAR - df["Year_Built"]
        ).clip(lower=0)

    # 7. Calculate price per square foot
    # Price is in lakhs, so multiply by 100,000
    if {"Price_in_Lakhs", "Size_in_SqFt"}.issubset(df.columns):

        size = df["Size_in_SqFt"].where(
            df["Size_in_SqFt"] > 0
        )

        df["Price_per_SqFt"] = (
            df["Price_in_Lakhs"] * 100000 / size
        )

    # 8. Create school and hospital accessibility scores
    for col, new_col in [
        ("Nearby_Schools", "School_Density_Score"),
        ("Nearby_Hospitals", "Hospital_Access_Score")
    ]:

        if col in df.columns:

            minimum = df[col].min()
            maximum = df[col].max()

            if pd.notna(minimum) and maximum > minimum:

                df[new_col] = (
                    (df[col] - minimum)
                    / (maximum - minimum)
                ) * 100

            else:
                df[new_col] = 0.0

    # 9. Count listed amenities
    if "Amenities" in df.columns:

        df["Amenities_Count"] = df["Amenities"].apply(
            lambda value: len([
                item
                for item in str(value).split(",")
                if item.strip()
                and item.strip().lower() != "nan"
            ])
        )

    # 10. Create a five-year price scenario
    if "Price_in_Lakhs" in df.columns:

        df["Future_Price_5Y_Lakhs"] = (
            df["Price_in_Lakhs"]
            * (1 + ANNUAL_GROWTH_RATE)
            ** FORECAST_YEARS
        )

    # 11. Create the Good Investment label
    if "Price_per_SqFt" in df.columns:

        median_price = df["Price_per_SqFt"].median()

        df["Good_Investment"] = (
            df["Price_per_SqFt"] <= median_price
        ).astype(int)

    # 12. Save cleaned dataset
    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # 13. Print information
    print("Cleaned dataset shape:", df.shape)
    print(
        "Remaining missing values:",
        df.isna().sum().sum()
    )
    print(
        "Duplicate rows:",
        df.duplicated().sum()
    )
    print(
        "Cleaned data saved to:",
        OUTPUT_PATH
    )

    return df


def get_feature_columns(df):

    # Exclude ID, direct label inputs and prediction targets
    excluded = {
        "ID",
        "Locality",
        "Price_per_SqFt",
        "Good_Investment",
        "Future_Price_5Y_Lakhs"
    }

    return [
        col
        for col in df.columns
        if col not in excluded
    ]


# Run preprocessing when this file is executed directly
if __name__ == "__main__":

    df = load_and_preprocess_data()

    print("\nMissing values by column:")
    print(
        df.isna()
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    print("\nFirst five cleaned rows:")
    print(df.head())

    print("\nPreprocessing completed successfully!")
    print("Final shape:", df.shape)