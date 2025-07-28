import streamlit as st
from database import init_db, save_chat, get_history
from auth import login_user, register_user, reset_password
from queryHandler import handle_query
import pandas as pd

init_db()

st.set_page_config(page_title="AI Product Chatbot", layout="wide", page_icon="🤖")

# --- Dark mode style ---
dark_mode = """
    <style>
    body { background-color: #121212; color: #FFFFFF; }
    .stTextInput input, .stButton button, .stTextArea textarea {
        background-color: #333; color: white; border-radius: 8px;
    }
    </style>
"""
st.markdown(dark_mode, unsafe_allow_html=True)

st.title("🛍️ AI Shopping Assistant")

# --- Session State ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""

# --- Login/Register Section ---
if not st.session_state.authenticated:
    st.subheader("🔐 Login or Register")
    choice = st.radio("Select action", ["Login", "Register", "Forgot Password"])

    if choice == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if login_user(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("✅ Logged in successfully!")
            else:
                st.error("❌ Invalid username or password.")

    elif choice == "Register":
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        st.markdown("**Password must be at least 8 characters long and include uppercase, lowercase, digit, and special character.**")
        if st.button("Register"):
            success, message = register_user(new_user, new_pass)
            if success:
                st.success("✅ Registered! Please login now.")
            else:
                st.warning(f"⚠️ {message}")

    else:  # Forgot Password
        username = st.text_input("Username")
        new_pass = st.text_input("New Password", type="password")
        st.markdown("**Password must be at least 8 characters long and include uppercase, lowercase, digit, and special character.**")
        if st.button("Reset Password"):
            success, message = reset_password(username, new_pass)
            if success:
                st.success("✅ Password reset successfully! Please login.")
            else:
                st.error(f"❌ {message}")

# --- Main Chat UI ---
else:
    st.sidebar.success(f"👤 Welcome, {st.session_state.username}!")
    user_query = st.text_input("Ask me anything about products...")

    if st.button("Ask"):
        if user_query.strip():
            bot_reply = handle_query(user_query)
            save_chat(st.session_state.username, user_query, bot_reply)
            st.markdown("### 🤖 Response")
            st.markdown(bot_reply, unsafe_allow_html=True)

    with st.expander("📜 View Chat History"):
        history = get_history(st.session_state.username)
        if not history.empty:
            st.dataframe(history[['timestamp', 'user_query', 'bot_response']].sort_values("timestamp", ascending=False))
        else:
            st.info("No chat history found.")
