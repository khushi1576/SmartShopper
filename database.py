from sqlalchemy import create_engine, text
import pandas as pd

DB_URL = "postgresql://postgres:Khushi1510@localhost:5432/aiAssistantDb"
engine = create_engine(DB_URL)

def init_db():
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        );
        """))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            username TEXT,
            user_query TEXT,
            bot_response TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """))

def save_chat(username, user_query, bot_response):
    with engine.begin() as conn:
        conn.execute(text("""
        INSERT INTO chat_history (username, user_query, bot_response)
        VALUES (:username, :user_query, :bot_response)
        """), {"username": username, "user_query": user_query, "bot_response": bot_response})

def get_history(username):
    query = text("SELECT * FROM chat_history WHERE username = :username ORDER BY timestamp DESC")
    return pd.read_sql(query, engine, params={"username": username})
