import streamlit as st
from supabase import create_client
from datetime import datetime

def create_supabase_client():
    """
    Create and return a Supabase client instance using Streamlit secrets
    """
    try:
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_KEY"]
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        st.error(f"Failed to initialize Supabase client: {e}")
        st.stop()

def add_training_data():
    """Admin interface to add knowledge base data"""
    st.title("🧠 Knowledge Base Management")
    
    if not st.session_state.get("is_admin", False):
        st.error("Access denied. Admin rights required.")
        return
    
    # Create Supabase client
    supabase = create_supabase_client()
    
    with st.form("knowledge_base_form"):
        st.subheader("Add Degree Field Knowledge")
        
        degree_field = st.text_input("Degree Field (e.g., Biology, Computer Science)")
        skills = st.text_area("Skills gained in this program")
        related_fields = st.text_area("Jobs/Industries suitable for this degree")
        knowledge_areas = st.text_area("Core knowledge areas in this field")
        
        submitted = st.form_submit_button("Add to Knowledge Base")
        
        if submitted:
            try:
                # Insert data into field_knowledge_map
                data = {
                    "degree_field": degree_field,
                    "primary_skills": skills,
                    "related_fields": related_fields,
                    "knowledge_areas": knowledge_areas,
                    "created_at": datetime.now().isoformat()
                }
                
                response = supabase.table('field_knowledge_map').insert(data).execute()
                
                if hasattr(response, 'data'):  # Check if response has data attribute
                    st.success("Knowledge base updated successfully!")
                else:
                    st.error("Error updating knowledge base")
                    
            except Exception as e:
                st.error(f"Error updating knowledge base: {e}")

if __name__ == "__main__":
    add_training_data()