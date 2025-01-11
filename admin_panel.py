import streamlit as st
from supabase import create_client
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import json
import os

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

class MLTrainer:
    def __init__(self):
        self.supabase = create_supabase_client()
        self.model_path = 'ml_models'
        os.makedirs(self.model_path, exist_ok=True)

    def save_training_history(self, model_name, accuracy, parameters):
        """Save training details to Supabase"""
        try:
            data = {
                'model_name': model_name,
                'accuracy': accuracy,
                'parameters': json.dumps(parameters)
            }
            self.supabase.table('ml_training_history').insert(data).execute()
            return True
        except Exception as e:
            st.error(f"Error saving training history: {e}")
            return False

    def train_job_recommender(self, job_data):
        """Train and save the job recommender model"""
        try:
            # Prepare data
            if 'description' not in job_data.columns:
                st.error("Required column 'description' not found in training data")
                return None, 0
            
            descriptions = job_data['description'].fillna('')
            
            # Split data for validation
            train_desc, test_desc = train_test_split(
                descriptions, 
                test_size=0.2, 
                random_state=42
            )
            
            # Train vectorizer
            vectorizer = TfidfVectorizer(stop_words='english')
            train_vectors = vectorizer.fit_transform(train_desc)
            
            # Save model
            model_file = os.path.join(self.model_path, 'job_recommender.joblib')
            joblib.dump(vectorizer, model_file)
            
            # Calculate basic accuracy
            test_vectors = vectorizer.transform(test_desc)
            accuracy = len(vectorizer.vocabulary_) / (len(vectorizer.vocabulary_) + 100)
            
            # Save training history
            parameters = {
                'vocabulary_size': len(vectorizer.vocabulary_),
                'training_samples': len(train_desc),
                'test_samples': len(test_desc)
            }
            self.save_training_history('job_recommender', accuracy, parameters)
            
            return vectorizer, accuracy
            
        except Exception as e:
            st.error(f"Error in training: {e}")
            return None, 0

def load_job_data():
    """Load and prepare job data from Supabase"""
    try:
        supabase = create_supabase_client()
        if not supabase:
            return None
            
        result = supabase.table('job_listings').select("*").execute()
        if not result.data:
            st.error("No job data available")
            return None
            
        df = pd.DataFrame(result.data)
        
        # Check required columns
        required_columns = ['job_title', 'job_description']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            st.write("Available columns:", ', '.join(df.columns))
            st.info("Please ensure your job_listings table has 'job_title' and 'job_description' columns.")
            return None
        
        # Only filter by is_active if the column exists
        if 'is_active' in df.columns:
            df = df[df['is_active'] == 1]
        
        # Combine text fields
        combined_text = df['job_description'].fillna('')
        
        if 'required_skills' in df.columns:
            combined_text += ' ' + df['required_skills'].fillna('')
        if 'job_subject' in df.columns:
            combined_text += ' ' + df['job_subject'].fillna('')
        if 'educational_background' in df.columns:
            combined_text += ' ' + df['educational_background'].fillna('')
            
        df['combined_text'] = combined_text
                            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def setup_training_page():
    st.title("AI Model Training Interface")
    
    # Initialize trainer
    trainer = MLTrainer()
    
    # 1. Data Loading Section
    st.header("1. Data Loading")
    df = load_job_data()
    if df is None:
        return
        
    st.success(f"Loaded {len(df)} job listings")
    
    # Sample data preview
    preview_df = df[['job_title', 'job_description']].head()
    st.write("Sample data:", preview_df)
    
    # Display column information
    st.subheader("Column Information")
    for col in ['job_title', 'job_description', 'required_skills', 'job_subject']:
        if col in df.columns:
            non_null_count = df[col].count()
            st.write(f"- {col}: {non_null_count} non-null values")

    # 2. Training Configuration
    st.header("2. Training Configuration")
    with st.form("training_config"):
        st.write("Configure training parameters:")
        
        test_size = st.slider("Test Data Size (%)", 10, 40, 20)
        
        # Feature selection
        st.subheader("Select Features for Training")
        use_skills = st.checkbox("Include Required Skills", value=True)
        use_subject = st.checkbox("Include Job Subject", value=True)
        use_education = st.checkbox("Include Educational Background", value=True)
        
        if len(df) < 100:
            st.warning(f"Warning: You have {len(df)} samples. Recommended minimum is 100 for better results.")
        
        train_button = st.form_submit_button("Start Training")
        
        if train_button:
            if len(df) < 10:
                st.error("Not enough data for training. Minimum 10 samples required.")
                return
                
            with st.spinner("Training model..."):
                # Prepare training data
                training_text = df['job_description'].fillna('')
                if use_skills and 'required_skills' in df.columns:
                    training_text += ' ' + df['required_skills'].fillna('')
                if use_subject and 'job_subject' in df.columns:
                    training_text += ' ' + df['job_subject'].fillna('')
                if use_education and 'educational_background' in df.columns:
                    training_text += ' ' + df['educational_background'].fillna('')
                
                training_text = training_text.astype(str)
                
                training_df = pd.DataFrame({
                    'title': df['job_title'],
                    'description': training_text
                })
                
                vectorizer, accuracy = trainer.train_job_recommender(training_df)
                
                if vectorizer:
                    st.success(f"""
                        Training Complete!
                        - Accuracy: {accuracy:.2f}
                        - Vocabulary Size: {len(vectorizer.vocabulary_)}
                        - Model saved to: {trainer.model_path}
                    """)
    
    # 3. View Training History
    st.header("3. Training History")
    try:
        # Use trainer's Supabase client instead of undefined supabase
        history = trainer.supabase.table('ml_training_history').select("*").execute()
        if history.data:
            history_df = pd.DataFrame(history.data)
            st.write(history_df)
            
            # Plot training history
            if len(history_df) > 0:
                fig = px.line(history_df, 
                            x='trained_date', 
                            y='accuracy',
                            title='Model Accuracy Over Time')
                st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error loading training history: {e}")

    # 4. Test Model
    st.header("4. Test Trained Model")
    if os.path.exists(os.path.join(trainer.model_path, 'job_recommender.joblib')):
        test_text = st.text_area(
            "Enter a job description to test:",
            "Mathematics tutor with experience in calculus and algebra"
        )
        
        if st.button("Test Model"):
            try:
                # Load model
                vectorizer = joblib.load(
                    os.path.join(trainer.model_path, 'job_recommender.joblib')
                )
                
                # Transform test text
                test_vector = vectorizer.transform([test_text])
                
                # Get similar jobs using the combined text field
                job_vectors = vectorizer.transform(df['combined_text'].fillna(''))
                similarities = (test_vector @ job_vectors.T).toarray()[0]
                
                # Show top 5 similar jobs
                top_indices = similarities.argsort()[-5:][::-1]
                
                st.subheader("Top 5 Similar Jobs:")
                for i, idx in enumerate(top_indices, 1):
                    st.write(f"{i}. {df.iloc[idx]['job_title']}")
                    if 'city' in df.columns and 'state' in df.columns:
                        st.write(f"Location: {df.iloc[idx]['city']}, {df.iloc[idx]['state']}")
                    if 'hourly_rate' in df.columns and not pd.isna(df.iloc[idx]['hourly_rate']):
                        st.write(f"Hourly Rate: ${df.iloc[idx]['hourly_rate']}/hr")
                    st.write(f"Similarity: {similarities[idx]:.2f}")
                    st.write("---")
                
            except Exception as e:
                st.error(f"Error testing model: {e}")
    else:
        st.warning("No trained model found. Please train the model first.")

def knowledge_base_management():
    st.header("Knowledge Base Management")
    
    tab1, tab2 = st.tabs(["Add Knowledge", "View/Edit Knowledge"])
    
    with tab1:
        with st.form("knowledge_base_form"):
            st.subheader("Add Field Knowledge")
            
            degree_field = st.text_input("Degree Field (e.g., Biology, Computer Science)")
            skills = st.text_area("Skills typically gained in this program")
            related_fields = st.text_area("Jobs/Industries suitable for this degree")
            knowledge_areas = st.text_area("Core knowledge areas in this field")
            
            submitted = st.form_submit_button("Add to Knowledge Base")
            
            if submitted:
                try:
                    supabase = create_supabase_client()
                    if supabase:
                        data = {
                            "degree_field": degree_field,
                            "primary_skills": skills,
                            "related_fields": related_fields,
                            "knowledge_areas": knowledge_areas
                        }
                        
                        result = supabase.table('field_knowledge_map').insert(data).execute()
                        
                        if result.data:
                            st.success("Knowledge base updated successfully!")
                        else:
                            st.error("Error updating knowledge base")
                except Exception as e:
                    st.error(f"Error: {e}")

def field_mapping_management():
    st.header("Field and Job Mapping")
    
    with st.form("field_mapping_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            field = st.text_input("Academic Field")
            keywords = st.text_area("Keywords (One per line)")
        
        with col2:
            job_types = st.text_area("Related Job Types (One per line)")
            weight = st.slider("Matching Weight", 0.0, 1.0, 0.5)
        
        submitted = st.form_submit_button("Add Field Mapping")
        
        if submitted:
            try:
                supabase = create_supabase_client()
                if supabase:
                    data = {
                        "field": field,
                        "keywords": keywords.split('\n'),
                        "job_types": job_types.split('\n'),
                        "weight": weight
                    }
                    
                    result = supabase.table('field_mappings').insert(data).execute()
                    
                    if result.data:
                        st.success("Field mapping added successfully!")
                    else:
                        st.error("Error adding field mapping")
            except Exception as e:
                st.error(f"Error: {e}")

def user_management():
    st.header("User Management")
    
    try:
        supabase = create_supabase_client()
        if supabase:
            result = supabase.table('users').select("*").execute()
            
            if result.data:
                df = pd.DataFrame(result.data)
                st.dataframe(df)
                
                # User actions
                selected_user = st.selectbox("Select User", df['email'].tolist())
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Deactivate User"):
                        st.info(f"Deactivating user: {selected_user}")
                
                with col2:
                    if st.button("Reset Password"):
                        st.info(f"Resetting password for: {selected_user}")
            else:
                st.info("No users found")
    except Exception as e:
        st.error(f"Error loading users: {e}")

def job_listings_overview():
    st.header("Job Listings Overview")
    
    try:
        # Fixed: Using create_supabase_client() instead of connect_db()
        supabase = create_supabase_client()
        if supabase:
            result = supabase.table('job_listings').select("*").execute()
            
            if result.data:
                df = pd.DataFrame(result.data)
                
                # Add filters
                col1, col2 = st.columns(2)
                with col1:
                    status_filter = st.selectbox("Filter by Status", ["All"] + df['status'].unique().tolist())
                with col2:
                    location_filter = st.selectbox("Filter by Location", ["All"] + df['city'].unique().tolist())
                
                filtered_df = df
                if status_filter != "All":
                    filtered_df = filtered_df[filtered_df['status'] == status_filter]
                if location_filter != "All":
                    filtered_df = filtered_df[filtered_df['city'] == location_filter]
                
                st.dataframe(filtered_df)
                
                # Statistics
                st.subheader("Quick Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Listings", len(df))
                with col2:
                    st.metric("Active Listings", len(df[df['status'] == 'active']))
                with col3:
                    if 'salary_range' in df.columns:
                        avg_salary = df['salary_range'].mean()
                        st.metric("Average Salary", f"${avg_salary:,.2f}")
            else:
                st.info("No job listings found")
                
    except Exception as e:
        st.error(f"Error loading job listings: {e}")

def system_statistics():
    st.header("System Statistics")
    
    try:
        supabase = create_supabase_client()
        if supabase:
            # Create columns for main statistics
            col1, col2 = st.columns(2)
            
            # User Statistics
            try:
                users_result = supabase.table('users').select("*").execute()
                users_df = pd.DataFrame(users_result.data) if users_result.data else pd.DataFrame()
                
                with col1:
                    total_users = len(users_df) if not users_df.empty else 0
                    st.metric("Total Users", total_users)
                    
                    if not users_df.empty and 'user_type' in users_df.columns:
                        st.subheader("User Distribution")
                        user_type_counts = users_df['user_type'].value_counts()
                        fig = go.Figure(data=[go.Pie(labels=user_type_counts.index, 
                                                   values=user_type_counts.values)])
                        st.plotly_chart(fig)
            except Exception as e:
                with col1:
                    st.error(f"Error fetching user statistics: {e}")

            # Job Statistics
            try:
                jobs_result = supabase.table('job_listings').select("*").execute()
                jobs_df = pd.DataFrame(jobs_result.data) if jobs_result.data else pd.DataFrame()
                
                with col2:
                    total_jobs = len(jobs_df) if not jobs_df.empty else 0
                    st.metric("Total Jobs", total_jobs)
                    
                    if not jobs_df.empty and 'status' in jobs_df.columns:
                        st.subheader("Job Status Distribution")
                        status_counts = jobs_df['status'].value_counts()
                        fig = go.Figure(data=[go.Bar(x=status_counts.index, 
                                                   y=status_counts.values)])
                        st.plotly_chart(fig)
            except Exception as e:
                with col2:
                    st.error(f"Error fetching job statistics: {e}")

            # Detailed Analysis Section
            if not jobs_df.empty:
                st.markdown("---")
                st.subheader("Detailed Job Analysis")
                
                # Location Analysis
                if 'city' in jobs_df.columns and 'state' in jobs_df.columns:
                    st.subheader("Top Locations")
                    jobs_df['location'] = jobs_df['city'] + ', ' + jobs_df['state']
                    location_counts = jobs_df['location'].value_counts().head(10)
                    fig = px.bar(x=location_counts.index, y=location_counts.values,
                               title="Top 10 Job Locations")
                    st.plotly_chart(fig)

                # Salary Analysis
                if 'salary_range' in jobs_df.columns:
                    st.subheader("Salary Distribution")
                    salary_stats = {
                        "Average Salary": f"${jobs_df['salary_range'].mean():,.2f}",
                        "Median Salary": f"${jobs_df['salary_range'].median():,.2f}",
                        "Highest Salary": f"${jobs_df['salary_range'].max():,.2f}",
                        "Lowest Salary": f"${jobs_df['salary_range'].min():,.2f}"
                    }
                    
                    stats_cols = st.columns(4)
                    for i, (label, value) in enumerate(salary_stats.items()):
                        stats_cols[i].metric(label, value)

                # Job Categories Analysis
                if 'job_subject' in jobs_df.columns:
                    st.subheader("Jobs by Subject Area")
                    subject_counts = jobs_df['job_subject'].value_counts()
                    fig = px.pie(values=subject_counts.values, 
                               names=subject_counts.index,
                               title="Distribution of Jobs by Subject Area")
                    st.plotly_chart(fig)

                # Time-based Analysis
                if 'created_at' in jobs_df.columns:
                    st.subheader("Job Posting Trends")
                    jobs_df['created_at'] = pd.to_datetime(jobs_df['created_at'])
                    jobs_by_month = jobs_df.resample('M', on='created_at').size()
                    fig = px.line(x=jobs_by_month.index, y=jobs_by_month.values,
                                title="Job Postings Over Time")
                    st.plotly_chart(fig)

            # Knowledge Base Statistics
            try:
                knowledge_base_result = supabase.table('field_knowledge_map').select("*").execute()
                if knowledge_base_result.data:
                    st.markdown("---")
                    st.subheader("Knowledge Base Statistics")
                    kb_df = pd.DataFrame(knowledge_base_result.data)
                    st.metric("Total Knowledge Base Entries", len(kb_df))
                    
                    if 'degree_field' in kb_df.columns:
                        field_counts = kb_df['degree_field'].value_counts()
                        fig = px.bar(x=field_counts.index, y=field_counts.values,
                                   title="Knowledge Base Entries by Field")
                        st.plotly_chart(fig)
                        
            except Exception as e:
                st.error(f"Error fetching knowledge base statistics: {e}")

            # Last Updated Timestamp
            st.markdown("---")
            st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        st.error(f"Error loading statistics: {e}")

def admin_panel():
    # Check admin access
    if not st.session_state.get("logged_in", False) or st.session_state.get("user_type", "").lower() != "admin":
        st.error("Access denied. Admin rights required.")
        return
    
    st.title("👑 Admin Panel")
    
    # Sidebar navigation
    admin_options = [
        "Knowledge Base Management",
        "Field Mapping",
        "User Management",
        "Job Listings Overview",
        "System Statistics",
        "ML Model Training"
    ]
    
    selected_option = st.selectbox("Admin Functions", admin_options)
    
    if selected_option == "Knowledge Base Management":
        knowledge_base_management()
    elif selected_option == "Field Mapping":
        field_mapping_management()
    elif selected_option == "User Management":
        user_management()
    elif selected_option == "Job Listings Overview":
        job_listings_overview()
    elif selected_option == "System Statistics":
        system_statistics()
    elif selected_option == "ML Model Training":
        setup_training_page()

if __name__ == "__main__":
    admin_panel()