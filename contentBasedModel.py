import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine #type:ignore
import joblib
import os

# Function to load and clean data from PostgreSQL
def load_data_from_postgre():
    engine = create_engine("postgresql+psycopg2://postgres:Khushi1510@localhost:5432/aiAssistantDb")
    query = "SELECT product_id, product_name, category, about_product FROM amazon_reviews;"
    df = pd.read_sql(query, engine)

    # Clean and validate
    df.columns = [col.strip() for col in df.columns]
    required_cols = ['product_id', 'product_name', 'category', 'about_product']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    df['product_name'] = df['product_name'].fillna('')
    df['category'] = df['category'].fillna('')
    df['about_product'] = df['about_product'].fillna('')

    # Drop duplicates
    df = df.drop_duplicates(subset='product_id', keep='first').reset_index(drop=True)

    return df

# Function to build and save model
def build_and_save_model():
    try:
        df = load_data_from_postgre()

        # Combine relevant text fields
        df['combined_features'] = df['product_name'] + ' ' + df['category'] + ' ' + df['about_product']

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

        # Compute cosine similarity matrix
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Ensure shape consistency
        if cosine_sim.shape[0] != df.shape[0]:
            raise ValueError("Mismatch in cosine similarity matrix and DataFrame shape.")

        # Package the model components
        model_data = {
            'vectorizer': vectorizer,
            'cosine_sim': cosine_sim,
            'df': df
        }

        # Save to file
        os.makedirs("models", exist_ok=True)
        joblib.dump(model_data, 'models/content_based_model.pkl')
        print("✅ Model saved successfully as 'models/content_based_model.pkl'")

    except Exception as e:
        print("❌ Error building or saving the model:", e)

if __name__ == '__main__':
    build_and_save_model()
