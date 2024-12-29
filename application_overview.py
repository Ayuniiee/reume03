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
            .from_("job_applications")
            .select("id as application_id, job_listings(job_title, job_subject), users(full_name), resume_path, status")
            .eq("job_listings.parent_email", user_email)
            .execute()
        )
        return response.get("data", [])
    except Exception as e:
        st.error(f"Error fetching applications: {e}")
        return []

def update_application_status(application_id, status):
    """Update the status of an application"""
    supabase = create_supabase_client()
    try:
        response = (
            supabase
            .from_("job_applications")
            .update({"status": status, "updated_at": datetime.now().isoformat()})
            .eq("id", application_id)
            .execute()
        )
        return response.get("status_code") == 204
    except Exception as e:
        st.error(f"Error updating application status: {e}")
        return False

def download_resume(resume_path):
    """Handle resume download"""
    try:
        # Simulate downloading from Supabase Storage
        supabase = create_supabase_client()
        response = supabase.storage.from_("resumes").download(resume_path)
        return response.content
    except Exception as e:
        st.error(f"Error downloading resume: {e}")
        return None

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
            app_id = app["application_id"]
            job_title = app["job_listings"]["job_title"]
            job_subject = app["job_listings"]["job_subject"]
            full_name = app["users"]["full_name"]
            resume_path = app.get("resume_path")
            status = app.get("status", "Pending")

            with st.container():
                col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 3])

                with col1:
                    st.write(f"**Name:** {full_name}")
                with col2:
                    st.write(f"**Job Title:** {job_title}")
                with col3:
                    st.write(f"**Subject:** {job_subject}")
                with col4:
                    if resume_path:
                        resume_content = download_resume(resume_path)
                        if resume_content:
                            st.download_button(
                                label="Download Resume",
                                data=resume_content,
                                file_name=f"resume_{full_name}.pdf",
                                mime="application/pdf",
                                key=f"download_{app_id}"
                            )
                with col5:
                    st.write(f"**Status:** {status}")
                    if status == 'Pending':
                        if st.button("Accept", key=f"accept_{app_id}"):
                            if update_application_status(app_id, "Accepted"):
                                st.success("Application accepted!")
                                st.experimental_rerun()
                        if st.button("Reject", key=f"reject_{app_id}"):
                            if update_application_status(app_id, "Rejected"):
                                st.success("Application rejected!")
                                st.experimental_rerun()
                st.divider()
    else:
        st.info("No applications found for your jobs.")

def main():
    application_overview()

if __name__ == "__main__":
    main()