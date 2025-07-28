# import bcrypt #type: ignore
# from sqlalchemy import text #type: ignore
# from database import engine

# def hash_password(password):
#     return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# def verify_password(password, hashed):
#     return bcrypt.checkpw(password.encode(), hashed.encode())

# def register_user(username, password):
#     with engine.begin() as conn:
#         conn.execute(text("""
#         INSERT INTO users (username, password)
#         VALUES (:username, :password)
#         """), {"username": username, "password": hash_password(password)})

# def login_user(username, password):
#     with engine.begin() as conn:
#         result = conn.execute(text("SELECT password FROM users WHERE username = :username"), {"username": username})
#         row = result.fetchone()
#         if row:
#             return verify_password(password, row[0])
#     return False



import bcrypt
from sqlalchemy import text
from database import engine

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

def is_valid_password(password):
    # Password must be at least 8 characters, contain uppercase, lowercase, digit, and special character
    import re
    pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
    return re.match(pattern, password)

def register_user(username, password):
    if not is_valid_password(password):
        return False, "Password must be at least 8 characters long and include uppercase, lowercase, digit, and special character."
    with engine.begin() as conn:
        result = conn.execute(text("SELECT username FROM users WHERE username = :username"), {"username": username})
        if result.fetchone():
            return False, "Username already exists."
        conn.execute(text("""
        INSERT INTO users (username, password)
        VALUES (:username, :password)
        """), {"username": username, "password": hash_password(password)})
    return True, "User registered successfully."

def login_user(username, password):
    with engine.begin() as conn:
        result = conn.execute(text("SELECT password FROM users WHERE username = :username"), {"username": username})
        row = result.fetchone()
        if row and verify_password(password, row[0]):
            return True
    return False

def reset_password(username, new_password):
    if not is_valid_password(new_password):
        return False, "Password must be at least 8 characters long and include uppercase, lowercase, digit, and special character."
    with engine.begin() as conn:
        result = conn.execute(text("SELECT username FROM users WHERE username = :username"), {"username": username})
        if not result.fetchone():
            return False, "Username does not exist."
        conn.execute(text("""
        UPDATE users SET password = :password WHERE username = :username
        """), {"username": username, "password": hash_password(new_password)})
    return True, "Password reset successfully."
