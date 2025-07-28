import pandas as pd

df = pd.read_csv("cleaned_amazon.csv")

def handle_query(query):
    query = query.lower()
    results = df.copy()

    if "phone" in query:
        results = results[results['product_name'].str.contains("phone", case=False)]

    if "laptop" in query:
        results = results[results['product_name'].str.contains("laptop", case=False)]

    if "under" in query:
        price = int(''.join(filter(str.isdigit, query.split("under")[1])))
        results = results[results['discounted_price'] < price]

    if "above" in query:
        price = int(''.join(filter(str.isdigit, query.split("above")[1])))
        results = results[results['discounted_price'] > price]

    if "between" in query:
        parts = [int(s) for s in query.replace("₹", "").split() if s.isdigit()]
        if len(parts) == 2:
            results = results[(results['discounted_price'] >= parts[0]) & (results['discounted_price'] <= parts[1])]

    if "rating" in query or "best" in query or "top" in query:
        results = results.sort_values(by='rating', ascending=False)

    if "cheapest" in query:
        results = results.sort_values(by='discounted_price').head(1)

    if "most expensive" in query:
        results = results.sort_values(by='discounted_price', ascending=False).head(1)

    if "most reviewed" in query:
        results = results.sort_values(by='rating_count', ascending=False).head(1)

    # Limit result
    top_result = results.head(5)

    # Format
    output = ""
    for _, row in top_result.iterrows():
        output += f"**{row['product_name']}**\n"
        output += f"💰 ₹{row['discounted_price']} | ⭐ {row['rating']} ({row['rating_count']} reviews)\n"
        output += f"[🔗 View Product]({row['product_link']})\n\n"
    return output if output else "Sorry, I couldn't find matching products."
