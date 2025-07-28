from flask import Flask, render_template, request, jsonify
from PIL import Image
from io import BytesIO
import joblib
import pandas as pd
import requests
from base64 import b64encode
from sqlalchemy import create_engine #type:ignore

app = Flask(__name__)

# Load models
content_model = joblib.load('models/content_Based_Model.pkl')
hybrid_model = joblib.load('models/hybrid_model.pkl')
item_model = joblib.load('models/item_based_model.pkl')
sentiment_df = joblib.load('models/sentiment_results.pkl')

# ---------- Utility Functions ----------

# Function to load data from PostgreSQL
def load_data_from_postgre():
    engine = create_engine("postgresql+psycopg2://postgres:Khushi1510@localhost:5432/aiAssistantDb")
    query = "SELECT product_id, product_name, category, about_product, img_link FROM amazon_reviews;"
    df = pd.read_sql(query, engine)
    return df

# Load product data from PostgreSQL
df_content = load_data_from_postgre()

def show_product_image(product_id):
    try:
        # Fetch the image URL from the PostgreSQL data (stored in the img_link column)
        img_url = df_content[df_content['product_id'] == product_id]['img_link'].values[0]
        return img_url  # Directly return the URL of the image
    except Exception as e:
        print(f"Error fetching image for product_id {product_id}: {e}")
        return None

def get_product_sentiment_info(product_id):
    product = sentiment_df[sentiment_df['product_id'] == product_id]
    if product.empty:
        return None, f"❌ Product ID '{product_id}' not found."

    sentiment_info = product[['product_id', 'product_name', 'sentiment_label', 'sentiment_score', 'rating', 'review_content']].drop_duplicates(subset=['review_content'])

    most_common_sentiment = sentiment_info['sentiment_label'].mode()[0]
    review_count = len(sentiment_info)
    sentiment_by_rating = sentiment_info.groupby('rating')['sentiment_label'].value_counts().unstack().fillna(0).to_dict()
    first_sentiment_score_by_rating = sentiment_info.groupby('rating').first()['sentiment_score'].to_dict()

    summary = {
        "Product ID": product_id,
        "Product Name": product['product_name'].iloc[0],
        "Most Common Sentiment": most_common_sentiment,
        "Total Reviews": review_count,
        "Sentiment Breakdown by Rating": sentiment_by_rating,
        "First Sentiment Score by Rating": first_sentiment_score_by_rating
    }

    return sentiment_info, summary

def get_recs(product_id):
    def get_content():
        sim = content_model['cosine_sim']
        if product_id not in df_content['product_id'].values: return []
        idx = df_content[df_content['product_id'] == product_id].index[0]
        sims = sorted(list(enumerate(sim[idx])), key=lambda x: x[1], reverse=True)[1:6]
        return [(df_content.iloc[i[0]]['product_id'], df_content.iloc[i[0]]['product_name']) for i in sims]

    def get_item():
        if product_id not in item_model['interaction_matrix'].columns: return []
        sims = item_model['item_similarity_df'][product_id].sort_values(ascending=False)
        ids = sims[sims.index != product_id].head(5).index.tolist()
        return [(pid, df_content[df_content['product_id'] == pid]['product_name'].values[0]) for pid in ids]

    def get_sentiment():
        s = sentiment_df[sentiment_df['product_id'] == product_id]
        if s.empty: return "No sentiment data available."
        return f"Sentiment: {s['sentiment_label'].mode()[0]}\nAverage Score: {s['sentiment_score'].mean():.2f}"

    return get_content(), get_item(), get_content()[:3] + get_item()[:2], get_sentiment()

# ---------- Flask Routes ----------

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        product_id = request.form['product_id'].strip()
        if not product_id:
            return jsonify({'error': 'Please enter a Product ID.'}), 400

        content_recs, item_recs, hybrid_recs, sentiment = get_recs(product_id)
        sentiment_info, sentiment_summary = get_product_sentiment_info(product_id)

        img_url = show_product_image(product_id)

        return render_template('index.html',
                               sentiment=sentiment,
                               sentiment_info=sentiment_info,
                               sentiment_summary=sentiment_summary,
                               product_id=product_id,
                               content_recs=content_recs,
                               item_recs=item_recs,
                               hybrid_recs=hybrid_recs,
                               img_url=img_url)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
