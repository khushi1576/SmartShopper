import pandas as pd

# Load the dataset
df = pd.read_csv("amazon.csv")

# Clean 'discounted_price' and 'actual_price' → remove ₹ and commas
df['discounted_price'] = (
    df['discounted_price']
    .str.replace('₹', '', regex=False)
    .str.replace(',', '', regex=False)
    .astype(float)
)

df['actual_price'] = (
    df['actual_price']
    .str.replace('₹', '', regex=False)
    .str.replace(',', '', regex=False)
    .astype(float)
)

# Clean 'discount_percentage' → remove '%' and convert to float
df['discount_percentage'] = (
    df['discount_percentage']
    .str.replace('%', '', regex=False)
    .astype(float)
)

# Convert 'rating' to float (some may be NaN or blank)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Clean 'rating_count' → remove commas and convert to int
df['rating_count'] = (
    df['rating_count']
    .str.replace(',', '', regex=False)
)
df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce').fillna(0).astype(int)

# Save cleaned data to a new CSV
df.to_csv("cleaned_amazon.csv", index=False)
print("✅ Cleaned data saved as 'cleaned_amazon.csv'")
