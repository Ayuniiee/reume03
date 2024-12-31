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
        </style>
    """, unsafe_allow_html=True)

    # Header Section
    st.markdown("<h1 style='text-align: center; color: #0066cc;'>Guidelines EduResume</h1>", unsafe_allow_html=True)
    
    # Main Description
    st.markdown("""
        <div class='about-section'>
            <h2>Welcome to EduResume 👋</h2>
            <p style='font-size: 1.1rem; line-height: 1.6;'>
                EduResume is your dedicated platform for educational and professional growth. We help educators and professionals 
                connect with the right opportunities while simplifying the application process. Let's make your career journey smoother and more rewarding!
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Guidelines Section
    st.markdown("<h3 style='margin-top: 2rem;'>How to Use EduResume 🛠️</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class='feature-card'>
                <h4>📝 Upload Your Resume</h4>
                <ul style='list-style-type: none; padding-left: 0;'>
                    <li>✓ Click on the "Upload Resume" button.</li>
                    <li>✓ Select your PDF resume file.</li>
                    <li>✓ Ensure your resume contains clear and updated information.</li>
                </ul>
            </div>
            
            <div class='feature-card'>
                <h4>🔍 Analyze Your Resume</h4>
                <ul style='list-style-type: none; padding-left: 0;'>
                    <li>✓ View the detailed analysis of your resume.</li>
                    <li>✓ Check the skills and experience detected.</li>
                    <li>✓ Get a score and improvement suggestions.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='feature-card'>
                <h4>💼 Explore Job Recommendations</h4>
                <ul style='list-style-type: none; padding-left: 0;'>
                    <li>✓ View personalized job suggestions based on your skills.</li>
                    <li>✓ Filter jobs by relevance and preferences.</li>
                    <li>✓ Apply directly to positions through the platform.</li>
                </ul>
            </div>
            
            <div class='feature-card'>
                <h4>📈 Improve Your Profile</h4>
                <ul style='list-style-type: none; padding-left: 0;'>
                    <li>✓ Follow the tips to enhance your resume.</li>
                    <li>✓ Update your skills and certifications regularly.</li>
                    <li>✓ Leverage additional resources to grow your career.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    # Contact Information
    st.markdown("""
        <div class='about-section' style='text-align: center;'>
            <h3>Get in Touch 📬</h3>
            <p>Have questions or suggestions? We'd love to hear from you!</p>
            <p>
                📧 Email: ayuniekhadijah@gmail.com<br>
                📞 Phone: 018-3802144<br>
                📍 Location: Universiti Pendidikan Sultan Idris
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <div style='text-align: center; margin-top: 3rem; padding: 1rem; background-color: #f8f9fa; border-radius: 8px;'>
            <p>© 2024 EduResume. All rights reserved.</p>
            <p style='font-size: 0.9rem; color: #666;'>
                Empowering educators and professionals, one resume at a time.
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    about_us()
