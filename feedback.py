import streamlit as st
from supabase import create_client
from datetime import datetime

def create_supabase_client():
    try:
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_KEY"]
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        st.error(f"Failed to initialize Supabase client: {e}")
        st.stop()

def star_rating_widget(label, max_stars=5, default_rating=3):
    """
    Custom star rating widget with enhanced styling
    """
    st.write(label)
    rating = st.session_state.get("current_rating", default_rating)
    
    # Create a container for better star alignment
    star_container = st.container()
    cols = star_container.columns(max_stars)
    
    # Enhanced star styling
    for i in range(max_stars):
        star_label = "★" if i < rating else "☆"
        if cols[i].button(star_label, key=f"star_{i+1}", help=f"Rate {i+1} stars"):
            st.session_state["current_rating"] = i + 1
            rating = i + 1

    return st.session_state.get("current_rating", default_rating)

def render_stars(rating):
    """
    Render star rating display with enhanced styling
    """
    full_star = "★"
    empty_star = "☆"
    stars = full_star * int(rating) + empty_star * (5 - int(rating))
    return f"<span style='color: gold; font-size: 20px;'>{stars}</span>"

def submit_feedback(supabase, full_name, user_email, rating, comment):
    """
    Submit feedback to the database
    """
    try:
        data = {
            "full_name": full_name,
            "user_email": user_email,
            "rating": rating,
            "comment": comment,
            "created_at": datetime.now().isoformat()
        }
        response = supabase.table("feedback").insert(data).execute()
        return hasattr(response, 'data')  # Check if response has data attribute
    except Exception as e:
        st.error(f"Error submitting feedback: {e}")
        return False

def fetch_feedbacks(supabase):
    """
    Fetch all feedbacks from the database
    """
    try:
        response = supabase.table("feedback").select("*").order('created_at', desc=True).execute()
        return response.data if hasattr(response, 'data') else []
    except Exception as e:
        st.error(f"Error fetching feedbacks: {e}")
        return []

def feedback():
    st.markdown("<h1 style='text-align: center; color: #0066cc;'>Feedback Portal</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>We value your thoughts and experiences</p>", unsafe_allow_html=True)

    # Create Supabase client
    supabase = create_supabase_client()

    # Feedback Submission Section
    if st.session_state.get("logged_in", False):
        st.subheader("📝 Share Your Experience")
        
        # Get user details from session state
        user_email = st.session_state.get("email", "")
        
        # Fetch full name from users table
        try:
            user_response = supabase.table("users").select("full_name").eq("email", user_email).single().execute()
            full_name = user_response.data.get('full_name', '') if user_response.data else ''
        except Exception as e:
            st.error(f"Error fetching user details: {e}")
            full_name = ""

        st.write(f"**👤 Name:** {full_name}")
        st.write(f"**📧 Email:** {user_email}")

        rating = star_rating_widget("Select your rating:", max_stars=5)
        comment = st.text_area("💭 Share Your Thoughts", height=100)

        if st.button("Submit Feedback"):
            if rating > 0:
                if submit_feedback(supabase, full_name, user_email, rating, comment):
                    st.success("🎉 Thank you for your valuable feedback!")
                    # Clear the form after successful submission
                    st.session_state["current_rating"] = 3  # Reset rating
                    st.rerun()  # Using st.rerun() instead of experimental_rerun
            else:
                st.warning("⚠️ Please select a rating.")
    else:
        st.info("👋 Please log in to share your experience!")

    # Fetch and display feedbacks
    st.markdown("---")
    st.subheader("📢 Community Feedback")
    
    feedbacks = fetch_feedbacks(supabase)
    
    if feedbacks:
        for feedback in feedbacks:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"<h3 style='margin:0;'>👤 {feedback['full_name']}</h3>", unsafe_allow_html=True)
            with col2:
                st.markdown(render_stars(feedback['rating']), unsafe_allow_html=True)
            st.markdown(f"<div>{feedback['comment']}</div>", unsafe_allow_html=True)
            created_at = datetime.fromisoformat(feedback['created_at'].replace('Z', '+00:00'))
            formatted_date = created_at.strftime("%Y-%m-%d %H:%M")
            st.markdown(f"<div style='color: #666;'>🕒 {formatted_date}</div>", unsafe_allow_html=True)
            st.markdown("---")
    else:
        st.info("📝 No feedbacks yet. Be the first to share your experience!")

if __name__ == "__main__":
    feedback()