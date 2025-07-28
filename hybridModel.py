import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine  # type: ignore

# Connect to PostgreSQL
def load_data_from_postgre():
    engine = create_engine("postgresql+psycopg2://postgres:Khushi1510@localhost:5432/aiAssistantDb")
    query = "SELECT product_id, product_name, category, about_product, rating, user_id FROM amazon_reviews;"
    return pd.read_sql(query, engine)

# Preprocess data (same as before)
def preprocess_data(df):
    # Fill missing values and combine relevant text features
    df['combined_features'] = (
        df['product_name'].fillna('') + ' ' +
        df['category'].fillna('') + ' ' +
        df['about_product'].fillna('')
    )

    # Convert the 'rating' column to numeric, forcing errors to NaN
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Drop rows with NaN in 'rating' after conversion (optional)
    df = df.dropna(subset=['rating'])
    
    return df

# Function to create the TF-IDF matrix and cosine similarity
def create_cosine_similarity(df):
    # Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')

    # Fit and transform the combined text features
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    # Calculate cosine similarity between all products
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Convert cosine similarity matrix to a DataFrame for easier handling
    cosine_sim_df = pd.DataFrame(cosine_sim, index=df['product_id'], columns=df['product_id'])

    return cosine_sim_df

# Function to create the item-based collaborative filtering model
def create_item_similarity(df):
    # Create interaction matrix: user-product interaction, where rating is the value
    interaction_matrix = df.pivot_table(index='user_id', columns='product_id', values='rating')
    interaction_matrix = interaction_matrix.fillna(0)

    # Calculate item similarity using cosine similarity
    item_similarity = cosine_similarity(interaction_matrix.T)

    # Convert similarity matrix to DataFrame for better visualization
    item_similarity_df = pd.DataFrame(item_similarity, index=interaction_matrix.columns, columns=interaction_matrix.columns)

    return item_similarity_df, interaction_matrix

# Function to save the model
def save_model(interaction_matrix, item_similarity_df, file_name='hybrid_model.pkl'):
    # Ensure the 'models' folder exists
    os.makedirs('models', exist_ok=True)

    # Save the model components (interaction matrix and item similarity)
    model_data = {
        'interaction_matrix': interaction_matrix,
        'item_similarity_df': item_similarity_df
    }
    file_path = os.path.join('models', file_name)
    joblib.dump(model_data, file_path)
    print(f"Model saved to {file_path}")

# Main function to load data, train the model, and save it
if __name__ == "__main__":
    # Load data from PostgreSQL
    df = load_data_from_postgre()

    # Preprocess data
    df = preprocess_data(df)

    # Create item-based similarity model
    item_similarity_df, interaction_matrix = create_item_similarity(df)

    # Save the model to the 'models' folder
    save_model(interaction_matrix, item_similarity_df)
