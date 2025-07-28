# sentiment.py

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sqlalchemy import create_engine # type:ignore

# Download required NLTK data if not already present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

try:
    SentimentIntensityAnalyzer().polarity_scores('test')
except LookupError:
    nltk.download('vader_lexicon')

# PostgreSQL connection details
username = 'postgres'
password = 'Khushi1510'
host = 'localhost'
port = '5432'
database = 'aiAssistantDb'
connection_string = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"

# Connect to PostgreSQL
engine = create_engine(connection_string)

# Load data from PostgreSQL
query = "SELECT * FROM amazon_reviews;"  # Adjust column names as needed
df = pd.read_sql(query, engine)

# Define text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)  # remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove non-letters
    text = text.lower()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Apply cleaning
df['cleaned_review'] = df['review_content'].apply(clean_text)

# Sentiment analysis
sid = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['cleaned_review'].apply(lambda x: sid.polarity_scores(x)['compound'])

# Sentiment classification
def classify_sentiment(score):
    if score > 0.1:
        return 'positive'
    elif score < -0.1:
        return 'negative'
    else:
        return 'neutral'

df['sentiment_label'] = df['sentiment_score'].apply(classify_sentiment)

# Function to get sentiment summary for a product
def get_product_sentiment_info(product_id):
    product = df[df['product_id'] == product_id].drop_duplicates(subset=['review_content'])  # Remove duplicate reviews
    if product.empty:
        return f"❌ Product ID '{product_id}' not found.", None

    sentiment_info = product[['product_id', 'product_name', 'sentiment_label', 'sentiment_score', 'rating', 'review_content']]
    
    # Remove duplicates from sentiment_info to avoid repeated reviews
    sentiment_info = sentiment_info.drop_duplicates(subset=['review_content'])

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


# Interactive CLI to test
if __name__ == "__main__":
    product_id_input = input("🔍 Enter product_id to analyze sentiment: ").strip()
    result, summary = get_product_sentiment_info(product_id_input)

    if isinstance(result, str):
        print(result)
    else:
        print("\n🧾 Sentiment Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")

        print("\n📋 Sample Reviews:")
        for idx, row in result.iterrows():
            print(f"- Rating: {row['rating']}, Sentiment: {row['sentiment_label']}, Score: {row['sentiment_score']}")
            print(f"  Review: {row['review_content']}\n")
