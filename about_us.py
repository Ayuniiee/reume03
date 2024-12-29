import streamlit as st

def about_us():
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .about-section {
            background-color: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        .feature-card {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .highlight {
            color: #0066cc;
            font-weight: bold;
        }
        .team-section {
            text-align: center;
            margin-top: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header Section
    st.markdown("<h1 style='text-align: center; color: #0066cc;'>About EduResume</h1>", unsafe_allow_html=True)
    
    # Main Description
    st.markdown("""
        <div class='about-section'>
            <h2>Welcome to EduResume 👋</h2>
            <p style='font-size: 1.1rem; line-height: 1.6;'>
                EduResume is your dedicated platform for educational and professional growth. We bridge the gap between 
                academic excellence and career opportunities, helping students and professionals showcase their potential 
                and find their dream positions in the educational sector.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Mission Statement
    st.markdown("""
        <div class='about-section'>
            <h3>Our Mission 🎯</h3>
            <p style='font-size: 1.1rem; line-height: 1.6;'>
                To empower educators and educational professionals by providing a streamlined platform for job searching,
                application management, and career development. We believe in making the job search process more efficient,
                transparent, and successful for everyone in the education sector.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Key Features Section
    st.markdown("<h3 style='margin-top: 2rem;'>What We Offer 🌟</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class='feature-card'>
                <h4>💼 Job Management</h4>
                <ul style='list-style-type: none; padding-left: 0;'>
                    <li>✓ Easy job search and filtering</li>
                    <li>✓ Application tracking system</li>
                    <li>✓ Status updates and notifications</li>
                </ul>
            </div>
            
            <div class='feature-card'>
                <h4>📝 Profile Building</h4>
                <ul style='list-style-type: none; padding-left: 0;'>
                    <li>✓ Professional profile creation</li>
                    <li>✓ Resume/CV management</li>
                    <li>✓ Skills showcase</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='feature-card'>
                <h4>🤝 Networking</h4>
                <ul style='list-style-type: none; padding-left: 0;'>
                    <li>✓ Connect with institutions</li>
                    <li>✓ Professional community</li>
                    <li>✓ Industry updates</li>
                </ul>
            </div>
            
            <div class='feature-card'>
                <h4>📈 Career Growth</h4>
                <ul style='list-style-type: none; padding-left: 0;'>
                    <li>✓ Professional development resources</li>
                    <li>✓ Career advice and tips</li>
                    <li>✓ Industry insights</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    # Contact Information
    st.markdown("""
        <div class='about-section' style='text-align: center;'>
            <h3>Get in Touch 📬</h3>
            <p>Have questions or suggestions? We'd love to hear from you!</p>
            <p>
                📧 Email: contact@eduresume.com<br>
                📞 Phone: (123) 456-7890<br>
                📍 Location: Education District, Knowledge City
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <div style='text-align: center; margin-top: 3rem; padding: 1rem; background-color: #f8f9fa; border-radius: 8px;'>
            <p>© 2024 EduResume. All rights reserved.</p>
            <p style='font-size: 0.9rem; color: #666;'>
                Making educational careers better, one application at a time.
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    about_us()