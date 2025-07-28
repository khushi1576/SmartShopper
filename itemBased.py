import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine  # type: ignore

def load_data_from_postgre():
    # Connect to PostgreSQL
    engine = create_engine("postgresql+psycopg2://postgres:Khushi1510@localhost:5432/aiAssistantDb")
    
    # Load the necessary data
    query = "SELECT user_id, product_id, rating, product_name FROM amazon_reviews;"
    df = pd.read_sql(query, engine)

    # Ensure ratings are numeric
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Fill missing ratings with 0 (optional, depending on business logic)
    df['rating'] = df['rating'].fillna(0)

    return df

def create_interaction_matrix(df):
    return df.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)

def calculate_item_similarity(interaction_matrix):
    return cosine_similarity(interaction_matrix.T)

def recommend_items_based_on_similarities(item_similarity_df, product_id, df, top_n=5):
    if product_id not in item_similarity_df.columns:
        raise ValueError(f"Product ID '{product_id}' not found in the dataset.")

    similar_scores = item_similarity_df[product_id].sort_values(ascending=False)
    similar_items = similar_scores[similar_scores.index != product_id].head(top_n)

    # Fetch product names
    recommendations = []
    for item_id in similar_items.index:
        product_name = df[df['product_id'] == item_id]['product_name'].values[0]
        recommendations.append((item_id, product_name))

    return recommendations

if __name__ == "__main__":
    try:
        df = load_data_from_postgre()
        interaction_matrix = create_interaction_matrix(df)
        item_similarity = calculate_item_similarity(interaction_matrix)

        item_similarity_df = pd.DataFrame(item_similarity, index=interaction_matrix.columns, columns=interaction_matrix.columns)

        product_id_input = input("Enter product_id to get recommendations: ").strip()

        recommended_items = recommend_items_based_on_similarities(item_similarity_df, product_id_input, df, top_n=5)

        print(f"\nTop 5 similar products to product ID '{product_id_input}':")
        for item_id, name in recommended_items:
            print(f"product_id: {item_id}, product_name: {name}")

    except Exception as e:
        print("Error:", e)
