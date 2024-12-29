import streamlit as st
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

def login():
    # Initialize session state if needed
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "page" not in st.session_state:
        st.session_state.page = "login"

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
    .input-container {{
        width: 50%;
        margin: 0 auto;
    }}
    .form-label {{
        color: white;
        font-size: 1rem;
        display: block;
        margin-bottom: 5px;
    }}
    .button-container {{
        text-align: center;
        margin-top: 20px;
    }}
    .sign-up-link {{
        color: white;
        font-size: 1rem;
        text-align: left;
        display: block;
        margin-top: 20px;
        margin-bottom: 5px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

    # Title Container
    st.markdown('<div class="title-container"><span class="welcome-text">Welcome to</span><span class="edu-title">EduResume</span></div>', unsafe_allow_html=True)

    # Input fields
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.markdown('<label class="form-label">Email</label>', unsafe_allow_html=True)
    email = st.text_input("", placeholder="Enter your email", key="login_email", label_visibility="collapsed")
    st.markdown('<label class="form-label">Password</label>', unsafe_allow_html=True)
    password = st.text_input("", placeholder="Enter your password", key="login_password", type="password", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # Button Container
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    
    if st.button("Log In", key="login_submit"):
        try:
            supabase = init_supabase()
            
            # Query user by email
            response = supabase.table('users').select("*").eq('email', email).execute()
            
            if response.data:
                user = response.data[0]
                stored_password = user['password']
                if isinstance(stored_password, str):
                    stored_password = stored_password.encode('utf-8')
                
                if bcrypt.checkpw(password.encode('utf-8'), stored_password):
                    st.success(f"Welcome, {email}!")
                    st.session_state["logged_in"] = True
                    st.session_state["email"] = email
                    st.session_state["user_type"] = user['user_type']
                    st.session_state["user_id"] = user['id']  # Note: Supabase uses lowercase 'id'
                    st.session_state["page"] = "home"
                    st.rerun()
                else:
                    st.error("Invalid password.")
            else:
                st.error("No user found with this email.")
                
        except Exception as e:
            st.error(f"Login error: {str(e)}")

    # Sign Up section
    st.markdown('<div class="sign-up-link">Don\'t have an account?</div>', unsafe_allow_html=True)
    
    if st.button("Sign Up", key="login_page_signup"):
        st.session_state["page"] = "signup"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)  # Close button container

if __name__ == "__main__":
    login()
