import streamlit as st
import base64
from supabase import create_client

def connect_supabase():
    try:
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_KEY"]
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        st.error(f"Failed to initialize Supabase client: {e}")
        st.stop()

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

def home():
    background_image = get_base64_image('./Logo/background2.jpg')
    right_image = get_base64_image('./Logo/elemenhome2.png')
    top_image = get_base64_image('./Logo/EduResume2.png')  # Add the path for the new image

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

        .stApp {{
            background-image: url("data:image/png;base64,{background_image}");
            background-size: cover;
            background-position: center;
            font-family: 'Poppins', sans-serif;
            height: 100vh;  /* Ensure full height */
            margin: 0;  /* Remove default margins */
        }}
        
        /* Remove padding from main container */
        .block-container {{
            padding: 0 !important;
            max-width: 100%;
        }}
        
        .content-container {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding: 4rem 6rem;
            color: white;
            min-height: 100vh;
            position: relative;
            z-index: 1;  /* Ensure content is above overlay */
        }}
        
        .text-container {{
            flex: 1;
            max-width: 60%;
            padding-top: 2rem;
            position: relative;
        }}
        
        .top-image-container {{
            display: flex;
            justify-content: flex-start;
            margin-bottom: 0.2rem;  
            margin-left: -2rem;  
            height: 150px;  
        }}
        
        .top-image-container img {{
            height: 100%;  
            width: auto;   
        }}

        .description {{
            font-size: 16px;
            line-height: 1.6;
            margin-bottom: 2rem;
            text-align: justify;  
        }}
        
        .image-container {{
            flex: 1;
            max-width: 40%;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 550px;  
        }}
        
        .image-container img {{
            width: auto;  
            height: 100%; 
            object-fit: contain;
        }}

        /* Add overlay to entire page */
        .content-container::before {{
            content: "";
            position: fixed;  /* Change to fixed to cover entire viewport */
            top: 0;
            left: 0;
            width: 100%;
            height: 100vh;  /* Cover full viewport height */
            background-color: rgba(0, 0, 0, 0.8);  
            z-index: 0;  /* Should be below the content and header */
        }}

        header {{
            display: none !important;
        }}

        /* Remove default streamlit margins */
        .css-1544g2n, .css-k1vhr4 {{
            margin: 0 !important;
            padding: 0 !important;
        }}
        
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Content container with top image, text, and right image
    st.markdown(
        f'''
        <div class="content-container">
            <div class="text-container">
                <div class="top-image-container">
                    <img src="data:image/png;base64,{top_image}" alt="Top Image" />  <!-- New Top Image -->
                </div>
                <div class="description">
                    Selamat datang ke Edu Resume – platform pintar yang direka khas untuk membantu anda membina dan meningkatkan peluang pekerjaan melalui Resume pendidikan anda! Di sini, kami memudahkan anda memilih keraya pilihan anda melalui pengalaman, kemahiran, dan kelayatan anda dalam bidang pendidikan. Hanya dengan beberapa langkah mudah, anda boleh memuat naik butiran akademik dan profesional anda, dan sistem kami akan memberi cadangan keryjaya di dalam menjadi tutor peribadi. Serta kami akan membuka lebih banyak peluang kerjaya pendidikan untuk masa depan yang cerah!
                </div>
            </div>
            <div class="image-container">
                <img src="data:image/png;base64,{right_image}" alt="Resume Illustration" />
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    home()
