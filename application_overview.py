import streamlit as st
from supabase import create_client
from datetime import datetime
import os

def create_supabase_client():
    try:
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_KEY"]
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        st.error(f"Failed to initialize Supabase client: {e}")
        st.stop()

def fetch_applications(user_email):
    """Fetch all applications for jobs posted by the parent"""
    supabase = create_supabase_client()
    try:
        response = (
            supabase
            .table("job_applications")
            .select(
                "id, status, resume_path, created_at, " +
                "job_listings(job_title, job_subject, parent_email), " +
                "users(full_name)"
            )
            .eq("job_listings.parent_email", user_email)
            .execute()
        )
        return response.data if hasattr(response, 'data') else []
    except Exception as e:
        st.error(f"Error fetching applications: {e}")
        return []

def update_application_status(application_id, status):
    """Update the status of an application"""
    supabase = create_supabase_client()
    try:
        response = (
            supabase
            .table("job_applications")
            .update({
                "status": status,
                "updated_at": datetime.now().isoformat()
            })
            .eq("id", application_id)
            .execute()
        )
        return hasattr(response, 'data')
    except Exception as e:
        st.error(f"Error updating application status: {e}")
        return False

def application_overview():
    st.title("Application Overview")

    if not st.session_state.get("logged_in"):
        st.warning("Please log in to view applications.")
        return

    user_email = st.session_state.get("email")
    if not user_email:
        st.error("Unable to retrieve your email. Please log in again.")
        return

    applications = fetch_applications(user_email)

    if applications:
        for app in applications:
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 3])

                with col1:
                    st.write(f"**Name:** {app['users']['full_name']}")
                with col2:
                    st.write(f"**Job Title:** {app['job_listings']['job_title']}")
                with col3:
                    st.write(f"**Subject:** {app['job_listings']['job_subject']}")
                with col4:
                    if app.get('resume_path'):
                        # Directly use the resume_path to create a download link
                        resume_path = app['resume_path']  # Adjust this if necessary to your local storage path
                        if os.path.exists(resume_path):
                            with open(resume_path, "rb") as f:
                                resume_content = f.read()
                            st.download_button(
                                label="Download Resume",
                                data=resume_content,
                                file_name=os.path.basename(resume_path),
                                mime="application/pdf",
                                key=f"download_{app['id']}"
                            )
                        else:
                            st.error("Resume file not found.")
                    else:
                        st.write("No resume uploaded")
                with col5:
                    status = app.get('status', 'Pending')
                    st.write(f"**Status:** {status}")
                    if status == 'Pending':
                        if st.button("Accept", key=f"accept_{app['id']}"):
                            if update_application_status(app['id'], "Accepted"):
                                st.success("Application accepted!")
                                st.rerun()
                        if st.button("Reject", key=f"reject_{app['id']}"):
                            if update_application_status(app['id'], "Rejected"):
                                st.success("Application rejected!")
                                st.rerun()
                st.divider()
    else:
        st.info("No applications found for your jobs.")

if __name__ == "__main__":
    application_overview()