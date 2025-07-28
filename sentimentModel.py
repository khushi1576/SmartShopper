import joblib
import pandas as pd
from sentiment import df  # Import the dataframe from the sentiment logic #type:ignore
import os

# Ensure the 'models' directory exists
os.makedirs('models', exist_ok=True)

# Save the processed data (df) inside the 'models' folder
joblib.dump(df, 'models/sentiment_results.pkl')
print("✅ Processed data saved as .pkl file inside 'models' folder.")

# Save the sentiment analyzer inside the 'models' folder
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
joblib.dump(sid, 'models/sentiment_analyzer.pkl')
print("✅ Sentiment model saved as .pkl file inside 'models' folder.")
