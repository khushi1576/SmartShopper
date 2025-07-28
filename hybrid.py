import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine  # type: ignore

# Load data from PostgreSQL
def load_data_from_postgre():
    engine = create_engine("postgresql+psycopg2://postgres:Khushi1510@localhost:5432/aiAssistantDb")
    query = "SELECT product_id, product_name, category, about_product, user_id, rating FROM amazon_reviews;"
    df = pd.read_sql(query, engine)

    # Clean and prepare
    df.columns = [col.strip() for col in df.columns]
    df['product_name'] = df['product_name'].fillna('')
    df['category'] = df['category'].fillna('')
    df['about_product'] = df['about_product'].fillna('')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])

    # Create combined text features
    df['combined_features'] = df['product_name'] + ' ' + df['category'] + ' ' + df['about_product']
    return df

# Build content-based similarity matrix
def build_content_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    return pd.DataFrame(cosine_similarity(tfidf_matrix), index=df['product_id'], columns=df['product_id'])

# Build item-based collaborative filtering similarity matrix
def build_collaborative_similarity(df):
    interaction_matrix = df.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)
    item_sim = cosine_similarity(interaction_matrix.T)
    return pd.DataFrame(item_sim, index=interaction_matrix.columns, columns=interaction_matrix.columns)

# Recommend items based on similarity
def recommend_items(product_id, item_sim_df, df, top_n=5):
    if product_id not in item_sim_df.columns:
        raise ValueError(f"Product ID '{product_id}' not found in similarity matrix.")
    
    scores = item_sim_df[product_id].drop(product_id).sort_values(ascending=False).head(top_n)
    return [
        (pid, df[df['product_id'] == pid]['product_name'].values[0]) for pid in scores.index
    ]

# Main execution
if __name__ == "__main__":
    try:
        df = load_data_from_postgre()

        # Build both models
        content_sim_df = build_content_similarity(df)
        item_sim_df = build_collaborative_similarity(df)

        product_id_input = input("Enter product_id to get recommendations: ").strip()

        print(f"\nTop 5 similar products based on Collaborative Filtering:")
        recommendations_cf = recommend_items(product_id_input, item_sim_df, df, top_n=5)
        for pid, name in recommendations_cf:
            print(f"product_id: {pid}, product_name: {name}")

        print(f"\nTop 5 similar products based on Content-Based Filtering:")
        recommendations_cb = recommend_items(product_id_input, content_sim_df, df, top_n=5)
        for pid, name in recommendations_cb:
            print(f"product_id: {pid}, product_name: {name}")

    except Exception as e:
        print("Error:", e)
