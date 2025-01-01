import streamlit as st
from supabase import create_client

def create_supabase_client():
    try:
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_KEY"]
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        st.error(f"Failed to initialize Supabase client: {e}")
        st.stop()

def job_list():
    # Page Title with Background Color
    st.markdown(
        "<h1 style='text-align: center; color: white; background-color: #4b79a1; padding: 10px; border-radius: 10px;'>📋 Manage Job Listings</h1>",
        unsafe_allow_html=True
    )

    # Check login status and user type
    if not st.session_state.get("logged_in"):
        st.warning("Please log in first.")
        return

    if st.session_state.get("user_type", "").lower() != "parent":
        st.error("Access denied. This page is for parents only.")
        return

    # Fetch parent's email from session state
    parent_email = st.session_state.get("email")
    supabase = create_supabase_client()

    # Fetch job listings for the parent
    try:
        response = (
            supabase
            .from_("job_listings")
            .select("*")
            .eq("parent_email", parent_email)
            .execute()
        )
        jobs = response.data
    except Exception as e:
        st.error(f"Error fetching job listings: {e}")
        return

    # Display job listings
    if not jobs:
        st.info("No job listings found. Use the Upload Job Listing page to add jobs.")
        return

    # Iterate over job listings
    for job in jobs:
        # Card-like container for each job listing
        st.markdown(
            f"""
            <div style="background-color: #f7f9fc; padding: 15px; border-radius: 10px; margin-bottom: 20px; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);">
                <h3 style="color: #4b79a1; margin-bottom: 5px;">{job['job_title']}</h3>
                <p style="margin: 5px 0;"><strong>Description:</strong> {job['job_description']}</p>
                <p style="margin: 5px 0;"><strong>Preferred Start Date:</strong> {job['preferred_start_date']}</p>
                <p style="margin: 5px 0;"><strong>Job Frequency:</strong> {job['job_frequency']}</p>
                <p style="margin: 5px 0;"><strong>Hourly Rate:</strong> ${job['hourly_rate']}</p>
                <p style="margin: 5px 0; color: {'green' if job['is_active'] else 'red'}; font-weight: bold;">
                    <strong>Status:</strong> {'Active' if job['is_active'] else 'Inactive'}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Buttons in Columns
        col1, col2 = st.columns(2)
        with col1:
            toggle_label = "Set Active" if not job.get('is_active', False) else "Set Inactive"
            toggle_status = st.button(
                label=toggle_label,
                key=f"toggle_{job['id']}",
                help=f"Click to {'activate' if not job.get('is_active', False) else 'deactivate'} this job."
            )
        with col2:
            delete_job = st.button(
                label="Delete Job",
                key=f"delete_{job['id']}",
                help="Click to permanently delete this job listing."
            )

        # Handle status toggle
        if toggle_status:
            new_status = not job.get('is_active', False)
            try:
                supabase.from_("job_listings").update({"is_active": new_status}).eq("id", job["id"]).execute()
                st.success(f"Job status updated to {'Active' if new_status else 'Inactive'}.")
                st.rerun()
            except Exception as e:
                st.error(f"Error updating job status: {e}")

        # Handle job deletion
        if delete_job:
            try:
                supabase.from_("job_listings").delete().eq("id", job["id"]).execute()
                st.success("Job deleted successfully.")
                st.rerun()
            except Exception as e:
                st.error(f"Error deleting job: {e}")

# Ensure this is only run when the script is directly executed
if __name__ == "__main__":
    job_list()