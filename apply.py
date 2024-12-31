import streamlit as st
import sys
import os
import base64
import json
import datetime
from supabase import create_client

def create_supabase_client():
    try:
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_KEY"]
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        st.error(f"Failed to initialize Supabase client: {e}")
        st.stop()

def initialize_session_state():
    """Initialize all session state variables"""
    if "initialized" not in st.session_state:
        session_defaults = {
            "initialized": True,
            "step": 1,
            "form_data": {},
            "availability_rows": [{"day": "Monday", "time": datetime.time(9, 0)}],
            "logged_in": False,
            "email": None,
            "user_id": None,
            "page": "login",
            "selected_job_title": "Unknown Job",
            "selected_job_subject": "Unknown Subject",
            "selected_job_location": "Unknown Location",
            "selected_job_skills": "Not specified",
        }
        for key, value in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

def save_uploaded_resume(uploaded_file):
    if uploaded_file is not None:
        os.makedirs('uploads', exist_ok=True)
        filename = f"resume_{os.urandom(8).hex()}.pdf"
        file_path = os.path.join('uploads', filename)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None

def create_download_link(file_path):
    """Create a download link for the uploaded PDF file"""
    try:
        with open(file_path, "rb") as f:
            pdf_data = f.read()
            b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
            
        file_name = os.path.basename(file_path)
        st.download_button(
            label="Download Resume",
            data=pdf_data,
            file_name=file_name,
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Error creating download link: {e}")

# Replace the display_pdf function with this one
def display_pdf(file_path):
    st.write("Resume uploaded successfully!")
    create_download_link(file_path)

def change_step(step_number):
    st.session_state.step = step_number

def add_availability_slot():
    st.session_state.availability_rows.append({
        "day": "Monday",
        "time": datetime.time(9, 0)
    })

def remove_availability_slot(idx):
    if len(st.session_state.availability_rows) > 1:  # Prevent removing last row
        st.session_state.availability_rows.pop(idx)

def submit_application(user_id):
    availability_json = json.dumps(st.session_state.form_data['availability'])
    supabase = create_supabase_client()

    try:
        # Get the job details from the database
        job_result = supabase.table('job_listings').select('id, job_title, city, state').eq('job_title', st.session_state.selected_job_title).execute()

        if not job_result.data:
            st.error("Could not find the selected job in the database.")
            return False

        job_id = job_result.data[0]['id']
        
        # Check for existing application
        existing_application = supabase.table('job_applications').select('id').eq('user_id', user_id).eq('job_id', job_id).execute()
        
        if existing_application.data:
            st.error("You have already applied for this job.")
            return False

        # Insert new application
        application_data = {
            "user_id": user_id,
            "job_id": job_id,
            "resume_path": st.session_state.form_data['resume_path'],
            "teaching_style": st.session_state.form_data['teaching_style'],
            "availability": availability_json,
            "is_confirmed": False,
            "created_at": datetime.datetime.now().isoformat(),
            "status": 'Pending'
        }

        response = supabase.table('job_applications').insert(application_data).execute()
        
        if hasattr(response, 'data'):
            st.success("Application submitted successfully!")
            return True
        else:
            st.error("Error submitting application.")
            return False

    except Exception as e:
        st.error(f"Error submitting application: {e}")
        return False

def apply():
    # Initialize session state at the start
    initialize_session_state()

    if not st.session_state.logged_in:
        st.error("Please log in to apply for a job.")
        return

    if not st.session_state.user_id:
        st.error("No user ID found. Please log in again.")
        return
        
    st.title("Job Application Form")
    st.write(f"Applying for: {st.session_state.selected_job_title}")

    # Step 1: Resume Upload
    if st.session_state.step == 1:
        st.header("Step 1: Upload Resume")
        uploaded_resume = st.file_uploader("Upload your Resume (PDF)", type=['pdf'])
        
        if uploaded_resume:
            resume_path = save_uploaded_resume(uploaded_resume)
            if resume_path:
                st.session_state.form_data['resume_path'] = resume_path
                st.subheader("Resume Preview")
                display_pdf(resume_path)
                st.button("Next Step", on_click=change_step, args=(2,))

    # Step 2: Additional Details
    elif st.session_state.step == 2:
        st.header("Step 2: Additional Details")
        
        teaching_style = st.text_area(
            "Describe your teaching technique (notes/exercises/gamification and etc)", 
            value=st.session_state.form_data.get('teaching_style', '')
        )

        st.subheader("Availability")
        for idx, row in enumerate(st.session_state.availability_rows):
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                row["day"] = st.selectbox(
                    f"Day {idx + 1}",
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                    key=f"day_{idx}"
                )
            with col2:
                row["time"] = st.time_input(f"Time {idx + 1}", value=row["time"], key=f"time_{idx}")
            with col3:
                if len(st.session_state.availability_rows) > 1:
                    st.button("Remove", key=f"remove_{idx}", on_click=remove_availability_slot, args=(idx,))

        st.button("Add Availability", on_click=add_availability_slot)

        st.session_state.form_data['teaching_style'] = teaching_style
        st.session_state.form_data['availability'] = [
            {"day": row["day"], "time": row["time"].strftime('%H:%M')}
            for row in st.session_state.availability_rows
        ]

        col1, col2 = st.columns(2)
        with col1:
            st.button("Back", on_click=change_step, args=(1,))
        with col2:
            if teaching_style:
                st.button("Review", on_click=change_step, args=(3,))

    # Step 3: Review and Submit
    elif st.session_state.step == 3:
        st.header("Step 3: Review Application")
        
        st.subheader("Resume")
        if 'resume_path' in st.session_state.form_data:
            display_pdf(st.session_state.form_data['resume_path'])
        
        st.subheader("Teaching Style")
        st.write(st.session_state.form_data['teaching_style'])

        st.subheader("Availability")
        for slot in st.session_state.form_data['availability']:
            st.write(f"- {slot['day']} at {slot['time']}")

        confirm = st.checkbox("I confirm all information is accurate")
        
        col1, col2 = st.columns(2)
        with col1:
            st.button("Back", on_click=change_step, args=(2,))
        with col2:
            if confirm:
                if st.button("Submit Application"):
                    if submit_application(st.session_state.user_id):
                        st.session_state.page = "applied_jobs"
                        st.rerun()

if __name__ == "__main__":
    apply()