import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine  # type: ignore

# Function to load data from PostgreSQL
def load_data_from_postgre():
    engine = create_engine("postgresql+psycopg2://postgres:Khushi1510@localhost:5432/aiAssistantDb")
    query = "SELECT product_id, product_name, category, about_product FROM amazon_reviews;"
    df = pd.read_sql(query, engine)

    # Clean and preprocess
    df.columns = [col.strip() for col in df.columns]
    required_cols = ['product_id', 'product_name', 'category', 'about_product']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    df['product_name'] = df['product_name'].fillna('')
    df['category'] = df['category'].fillna('')
    df['about_product'] = df['about_product'].fillna('')

    # 🔥 Drop duplicate product_ids
    df = df.drop_duplicates(subset='product_id', keep='first').reset_index(drop=True)

    return df

# Function to recommend similar items
def recommend_items_for_product(product_id, df, cosine_sim, top_n=5):
    if product_id not in df['product_id'].values:
        raise ValueError("Product ID not found in dataset.")

    index = df[df['product_id'] == product_id].index[0]
    similarity_scores = list(enumerate(cosine_sim[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Exclude the product itself
    top_similar = [score for score in similarity_scores if df.iloc[score[0]]['product_id'] != product_id]

    # Pick top N unique products
    seen_ids = set()
    unique_recs = []
    for i, _ in top_similar:
        pid = df.iloc[i]['product_id']
        if pid not in seen_ids:
            unique_recs.append(i)
            seen_ids.add(pid)
        if len(unique_recs) == top_n:
            break

    recommendations = df.iloc[unique_recs][['product_id', 'product_name']].reset_index(drop=True)
    return recommendations

# Main logic
if __name__ == '__main__':
    try:
        df = load_data_from_postgre()

        # Combine text features
        df['combined_features'] = (
            df['product_name'] + ' ' + df['category'] + ' ' + df['about_product']
        )

        # Vectorize text and compute similarity
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Get input
        product_id_input = input("Enter product_id to get recommendations: ").strip()

        # Generate and print recommendations
        recommended_items = recommend_items_for_product(product_id_input, df, cosine_sim, top_n=5)
        print("\nTop 5 similar products:")
        for _, row in recommended_items.iterrows():
            print(f"product_id: {row['product_id']}, product_name: {row['product_name']}")

    except Exception as e:
        print("Error:", e)
