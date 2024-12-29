import streamlit as st
from supabase import create_client
import pandas as pd

def get_db_connection():
    try:
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_KEY"]
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        st.error(f"Error connecting to Supabase: {e}")
        return None

def fetch_applied_jobs(user_id):
    supabase = get_db_connection()
    if not supabase:
        return []

    try:
        # Using Supabase's query builder to join tables and fetch data
        response = supabase.rpc(
            'get_applied_jobs',  # Name of the stored procedure we'll create
            {'user_id_param': user_id}
        ).execute()

        if response.data:
            return response.data
        return []
    except Exception as e:
        st.error(f"Error fetching jobs: {e}")
        return []

def main():
    # Set a background color and padding for the main container
    st.markdown("""
    <style>
        .main {
            background-color: #f0f4f8;  /* Light background color */
            padding: 20px;              /* Padding around the content */
            border-radius: 8px;         /* Rounded corners */
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("Applied Jobs 💼", anchor="top")

    # Check if user is logged in
    if not st.session_state.get("logged_in"):
        st.warning("Please log in to view your applied jobs.", icon="⚠️")
        return

    # Get the user ID from the session state
    user_id = st.session_state.get("user_id")
    if not user_id:
        st.error("Unable to retrieve your user ID. Please log in again.", icon="❌")
        return

    # Fetch applied jobs for the logged-in user
    jobs = fetch_applied_jobs(user_id)

    if jobs:
        # Prepare data for display
        job_data = []
        for index, job in enumerate(jobs, start=1):
            job_data.append({
                "No": index,
                "Job Title": job['job_title'],
                "Job Subject": job['job_subject'],
                "City": job['city'],
                "State": job['state'],
                "Job Frequency": job['job_frequency'],
                "Status": job['status'] or 'Pending'
            })

        # Convert to DataFrame and display with custom styling
        df = pd.DataFrame(job_data)
        
        # Display DataFrame with styled columns
        st.markdown("""
        <style>
            .dataframe th {
                background-color: #021659;
                color: white;
                padding: 10px;
            }
            .dataframe td {
                border: 1px solid #ddd;
                padding: 8px;
            }
            .stDataFrame {
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
        </style>
        """, unsafe_allow_html=True)

        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("No applied jobs found.", icon="🔍")

if __name__ == "__main__":
    main()
