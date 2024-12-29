import streamlit as st
import re
import bcrypt
import base64
from supabase import create_client

def init_supabase():
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    return create_client(supabase_url, supabase_key)

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

background_image = get_base64_image('./Logo/background1.png')

def is_valid_email(email):
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, email) is not None

def is_valid_password(password):
    return (
        len(password) >= 8 and 
        any(c.isupper() for c in password) and 
        any(c.islower() for c in password) and 
        any(c.isdigit() for c in password)
    )

def signup():
    st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lilita+One&display=swap');

    /* Background settings */
    .stApp {{
        background-image: url("data:image/png;base64,{background_image}");
        background-size: cover;
        background-position: center;
        font-family: sans-serif;
    }}
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5);
        z-index: 0;
        pointer-events: none;
    }}

    /* Custom styling */
    .title-container {{
        text-align: center;
        margin-bottom: 30px;
    }}
    .welcome-text {{
        font-size: 40px;
        font-weight: bold;
        color: white;
        text-shadow: 2px 2px 0 black, -2px -2px 0 black, 2px -2px 0 black, -2px 2px 0 black;
        margin-right: 10px;
    }}
    .edu-title {{
        font-size: 40px;
        font-weight: bold;
        color: #00BFFF;
        text-shadow: 2px 2px 0 black, -2px -2px 0 black, 2px -2px 0 black, -2px 2px 0 black;
    }}
    
    /* Style for form inputs */
    div[data-baseweb="input"] input,
    div[data-baseweb="select"] div[role="button"] {{
        background-color: rgba(255, 255, 255, 0.9) !important;
    }}
    
    /* Style for labels */
    .stSelectbox label,
    .stTextInput label {{
        color: white !important;
        font-size: 1rem !important;
        text-shadow: 1px 1px 2px black;
    }}
    
    /* Container for the form */
    .form-container {{
        background-color: rgba(0, 0, 0, 0.6);
        padding: 20px;
        border-radius: 10px;
        margin: 0 auto;
        max-width: 800px;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

    # Title Container
    st.markdown('<div class="title-container"><span class="welcome-text">Welcome to</span><span class="edu-title">EduResume</span></div>', unsafe_allow_html=True)

    # Form Container
    st.markdown('<div class="form-container">', unsafe_allow_html=True)

    # Create columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        email = st.text_input("Email", placeholder="Enter your email")
        username = st.text_input("Username", placeholder="Choose a username")
        full_name = st.text_input("Full Name", placeholder="Enter your full name")

    with col2:
        password = st.text_input("Password", placeholder="Create a password", type="password")
        confirm_password = st.text_input("Confirm Password", placeholder="Confirm your password", type="password")
        user_type = st.selectbox("User Type", ["", "User", "Parent"])

    # Signup button
    if st.button("Sign Up", key="signup_submit"):
        # Validation checks
        if not all([email, username, full_name, password, confirm_password, user_type]):
            st.error("All fields are required.")
            return
        
        if not is_valid_email(email):
            st.error("Please enter a valid email address.")
            return
        
        if not is_valid_password(password):
            st.error("Password must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, and one number.")
            return
        
        if password != confirm_password:
            st.error("Passwords do not match.")
            return

        try:
            supabase = init_supabase()
            
            # Check if email exists
            email_check = supabase.table('users').select("*").eq('email', email).execute()
            if email_check.data:
                st.error("Email already registered. Please use a different email.")
                return

            # Check if username exists
            username_check = supabase.table('users').select("*").eq('username', username).execute()
            if username_check.data:
                st.error("Username already taken. Please choose a different username.")
                return

            # Hash password
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            hashed_password_str = hashed_password.decode('utf-8')

            # Insert new user
            user_data = {
                'email': email,
                'username': username,
                'full_name': full_name,
                'password': hashed_password_str,
                'user_type': user_type
            }
            
            result = supabase.table('users').insert(user_data).execute()
            
            if result.data:
                st.success("Account created successfully!")
                st.session_state["page"] = "login"
                st.rerun()
            else:
                st.error("Failed to create account. Please try again.")

        except Exception as e:
            st.error(f"Registration error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)  # Close form container

if __name__ == "__main__":
    signup()
