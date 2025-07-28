import pandas as pd
from sqlalchemy import create_engine #type:ignore
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import os

# Function to load data from PostgreSQL
def load_data_from_postgre():
    # Create SQLAlchemy engine
    engine = create_engine("postgresql+psycopg2://postgres:Khushi1510@localhost:5432/aiAssistantDb")

    # Load required data from your actual table
    query = "SELECT user_id, product_id, rating FROM amazon_reviews;"  # Adjust query to match your actual table
    df = pd.read_sql(query, engine)

    # Clean column names (remove extra spaces if any)
    df.columns = [col.strip() for col in df.columns]

    # Convert 'rating' column to numeric and fill NaN with 0
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['rating'] = df['rating'].fillna(0)

    return df

# Function to create interaction matrix
def create_interaction_matrix(df):
    interaction_matrix = df.pivot_table(index='user_id', columns='product_id', values='rating')
    return interaction_matrix.fillna(0)

# Function to calculate item similarities
def calculate_item_similarity(interaction_matrix):
    return cosine_similarity(interaction_matrix.T)

# Function to recommend items based on similarity (for later use)
def recommend_items_based_on_similarities(item_similarity_df, product_id, top_n=5):
    if product_id not in item_similarity_df.columns:
        raise ValueError(f"Product ID '{product_id}' not found in the dataset.")
    similar_scores = item_similarity_df[product_id]
    similar_items = similar_scores.sort_values(ascending=False)
    similar_items = similar_items[similar_items.index != product_id].head(top_n)
    return similar_items.index.tolist()

# Save the model components to a .pkl file inside the 'models' folder
def save_model(interaction_matrix, item_similarity_df, file_name='item_based_model.pkl'):
    # Ensure the 'models' folder exists
    os.makedirs('models', exist_ok=True)
    
    # Save the model to the 'models' folder
    model_data = {'interaction_matrix': interaction_matrix, 'item_similarity_df': item_similarity_df}
    file_path = os.path.join('models', file_name)
    joblib.dump(model_data, file_path)
    print(f"Model saved to {file_path}")

# Main function to load data, train the model, and save it
if __name__ == "__main__":
    # Load data from PostgreSQL
    df = load_data_from_postgre()
    
    # Create interaction matrix
    interaction_matrix = create_interaction_matrix(df)
    
    # Calculate item similarities
    item_similarity = calculate_item_similarity(interaction_matrix)
    
    # Convert similarity matrix to DataFrame for easier visualization
    item_similarity_df = pd.DataFrame(item_similarity, index=interaction_matrix.columns, columns=interaction_matrix.columns)
    
    # Save model to the 'models' folder
    save_model(interaction_matrix, item_similarity_df)
    
    print("Model has been saved successfully!")
