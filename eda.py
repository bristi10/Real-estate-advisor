import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from feature_engineering import load_and_preprocess_data


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = load_and_preprocess_data()

print("\nAvailable columns:")
print(df.columns.tolist())


# --------------------------------------------------
# 2. PRICE TRENDS BY CITY
# --------------------------------------------------

avg_price_city = (
    df.groupby("City")["Price_in_Lakhs"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Price by City:")
print(avg_price_city)


plt.figure(figsize=(14, 7))

avg_price_city.plot(kind="bar")

plt.title("Average Property Price by City")
plt.xlabel("City")
plt.ylabel("Average Price (Lakhs)")
plt.xticks(rotation=90)
plt.tight_layout()

plt.show()


# --------------------------------------------------
# 3. AREA VS INVESTMENT RETURN
# --------------------------------------------------

# Calculate expected investment gain after 5 years
df["Investment_Return_Lakhs"] = (
    df["Future_Price_5Y_Lakhs"]
    - df["Price_in_Lakhs"]
)

correlation = df[
    ["Size_in_SqFt", "Investment_Return_Lakhs"]
].corr().iloc[0, 1]

print("\nArea vs Investment Return Correlation:")
print(correlation)


plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="Size_in_SqFt",
    y="Investment_Return_Lakhs",
    alpha=0.3
)

plt.title("Property Area vs Expected Investment Return")
plt.xlabel("Property Size (SqFt)")
plt.ylabel("Expected Investment Return (Lakhs)")
plt.tight_layout()

plt.show()


# --------------------------------------------------
# 4. SECURITY VS GOOD INVESTMENT
# --------------------------------------------------

security_investment = pd.crosstab(
    df["Security"],
    df["Good_Investment"],
    normalize="index"
) * 100

print("\nSecurity Level vs Good Investment (%):")
print(security_investment)


plt.figure(figsize=(10, 6))

security_investment.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6)
)

plt.title("Security Level vs Good Investment")
plt.xlabel("Security")
plt.ylabel("Percentage (%)")
plt.legend(
    ["Not Good Investment", "Good Investment"],
    title="Investment"
)

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


# --------------------------------------------------
# 5. INFRASTRUCTURE / ACCESSIBILITY VS FUTURE VALUE
# --------------------------------------------------

# Create a simple infrastructure score
# using schools, hospitals and public transport.

df["Infrastructure_Score"] = (
    df["School_Density_Score"]
    + df["Hospital_Access_Score"]
) / 2


# Convert public transport accessibility into numbers
transport_mapping = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

if df["Public_Transport_Accessibility"].dtype == "object":

    df["Transport_Score"] = (
        df["Public_Transport_Accessibility"]
        .astype(str)
        .str.strip()
        .map(transport_mapping)
    )

else:
    df["Transport_Score"] = pd.to_numeric(
        df["Public_Transport_Accessibility"],
        errors="coerce"
    )


# Fill missing transport scores
df["Transport_Score"] = df["Transport_Score"].fillna(
    df["Transport_Score"].median()
)


# Final infrastructure score
df["Infrastructure_Score"] = (
    df["Infrastructure_Score"]
    + (df["Transport_Score"] / 3 * 100)
) / 2


correlation_infrastructure = df[
    ["Infrastructure_Score", "Future_Price_5Y_Lakhs"]
].corr().iloc[0, 1]

print("\nInfrastructure Score vs Future Property Value Correlation:")
print(correlation_infrastructure)


plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="Infrastructure_Score",
    y="Future_Price_5Y_Lakhs",
    alpha=0.3
)

plt.title("Infrastructure Score vs Future Property Value")
plt.xlabel("Infrastructure Score")
plt.ylabel("Future Price After 5 Years (Lakhs)")
plt.tight_layout()

plt.show()


print("\nEDA completed successfully!")