import nltk
nltk.download('stopwords')
import streamlit as st
import pandas as pd
import base64
import random
import time
import datetime
import re
import os
from supabase import create_client
from streamlit_tags import st_tags
from Courses import resume_videos, interview_videos
from keywords import it_keywords, software_keywords, multimedia_keywords, science_keywords, math_keywords
import spacy

# Replace with your actual Supabase project details
SUPABASE_URL = "https://zccnwnfslnafqkwfynjg.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpjY253bmZzbG5hZnFrd2Z5bmpnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzUyMTc3NDEsImV4cCI6MjA1MDc5Mzc0MX0.NuDDOv7NabiRQywA58klp17As7FM-n4hZzNPW8vJb2Y"

# Initialize the Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
def setup_spacy():
    """Setup spaCy with proper model loading"""
    try:
        # Download the English model if not already present
        if not spacy.util.is_package("en_core_web_sm"):
            os.system('python -m spacy download en_core_web_sm')
        # Load the model properly
        return spacy.load("en_core_web_sm")
    except Exception as e:
        st.error(f"Error setting up spaCy: {str(e)}")
        return None

# Initialize spaCy
nlp = setup_spacy()

class CustomResumeParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.nlp = nlp
        
    def extract_text_from_pdf(self):
        """Extract text from PDF file"""
        try:
            from pdfminer.high_level import extract_text
            return extract_text(self.file_path)
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return None
            
    def get_extracted_data(self):
        """Extract all data from resume with enhanced error handling"""
        try:
            text = self.extract_text_from_pdf()
            if not text:
                return None

            extracted_skills = self.extract_skills(text)
            
            # Group skills by category for better organization
            categorized_skills = {
                'teaching': [],
                'subjects': [],
                'technical': [],
                'other': []
            }
            
            # Categorize extracted skills
            for skill in extracted_skills:
                if any(word in skill.lower() for word in ['teach', 'classroom', 'education', 'learning']):
                    categorized_skills['teaching'].append(skill)
                elif any(word in skill.lower() for word in ['english', 'malay', 'arabic', 'science', 'math']):
                    categorized_skills['subjects'].append(skill)
                elif any(word in skill.lower() for word in ['ms', 'computer', 'programming', 'analysis']):
                    categorized_skills['technical'].append(skill)
                else:
                    categorized_skills['other'].append(skill)

            data = {
                'name': self.extract_name(text),
                'email': self.extract_email(text),
                'mobile_number': self.extract_mobile_number(text),
                'skills': extracted_skills,
                'categorized_skills': categorized_skills,
                'no_of_pages': 1
            }
            return data

        except Exception as e:
            st.error(f"Error parsing resume: {str(e)}")
            return None
            
    def extract_name(self, text):
        """
        Enhanced name extraction with improved pattern matching and filtering
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        # Words that indicate we should skip this line
        # In the extract_name method, update the skip_words list:
        skip_words = [
            'resume', 'curriculum vitae', 'cv', 'highly', 'experienced', 'professional',
            'qualified', 'dedicated', 'motivated', 'kindergarten', 'teacher', 'tutor',
            'software', 'engineer', 'developer', 'senior', 'junior', 'application',
            'passionate', 'creative', 'innovative', 'address', 'email', 'phone',
            'contact', 'education', 'experience', 'skills', 'objective',
            'lot', 'kampung', 'jalan', 'road', 'street', 'avenue', 'lane',
            'kuala', 'terengganu', 'selangor', 'penang', 'melaka', 'johor',
            'block', 'unit', 'floor', 'apartment'
        ]
        
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            
            # Skip lines with common words that aren't names
            if any(word.lower() in line.lower() for word in skip_words):
                continue
                
            # Special check for names starting with Sharifah/Syed
            if line.startswith('Sharifah') or line.startswith('Syed'):
                words = line.split()
                if 2 <= len(words) <= 5:  # Name should be 2-5 words
                    if all(word[0].isupper() for word in words):  # All words should be capitalized
                        if not any(skip_word.lower() in line.lower() for skip_word in skip_words):
                            return line.strip()
        
        # If no Sharifah/Syed name found, try general name patterns
        malay_name_patterns = [
            r'^[A-Z][a-z]+\s+(?:bin|binti|bt|b\.)\s+[A-Z][a-z]+',
            r'^(?:Sharifah|Syed)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+(?:bin|binti|bt|b\.)\s+[A-Z][a-z]+)?'
        ]
        
        for line in lines[:5]:
            line = line.strip()
            if any(skip_word.lower() in line.lower() for skip_word in skip_words):
                continue
                
            for pattern in malay_name_patterns:
                match = re.match(pattern, line)
                if match:
                    return line.strip()
        
        return 'Name not found'
        
    def extract_email(self, text):
        """Extract email from text"""
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        emails = re.findall(email_pattern, text)
        return emails[0] if emails else 'Not found'
        
    def extract_mobile_number(self, text):
        """Extract mobile number from text"""
        phone_pattern = r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]'
        numbers = re.findall(phone_pattern, text)
        return numbers[0] if numbers else 'Not found'
        

    def extract_skills(self, text):
        """
        Enhanced skill extraction with better subject area matching
        """
        # Define comprehensive skill and subject keywords
        subject_areas = {
            'mathematics': [
                'mathematics', 'algebra', 'calculus', 'geometry', 'trigonometry',
                'statistics', 'probability', 'arithmetic', 'number theory', 'math',
                'mathematical', 'numeracy', 'quantitative', 'additional mathematics',
                'modern math'
            ],
            'science': [
                'science', 'physics', 'chemistry', 'biology', 'environmental science',
                'earth science', 'laboratory', 'scientific method', 'experiments',
                'stem', 'scientific', 'computer science', 'information technology',
                'science computer', 'applied science'
            ],
            'languages': [
                'english', 'malay', 'bahasa melayu', 'arabic', 'mandarin', 'chinese',
                'language arts', 'grammar', 'composition', 'literature', 'quran',
                'linguistics', 'english language', 'bahasa inggeris', 'tamil',
                'japanese', 'korean', 'french', 'german', 'spanish'
            ],
            'humanities': [
                'history', 'geography', 'economics', 'business studies', 'accounting',
                'commerce', 'economy', 'social studies', 'civics', 'moral education',
                'islamic studies', 'religious studies'
            ],
            'technical': [
                'computer science', 'information technology', 'programming',
                'web development', 'database', 'networking', 'cybersecurity',
                'digital technology', 'multimedia', 'graphic design',
                'engineering drawing', 'technical drawing'
            ],
            'vocational': [
                'home science', 'living skills', 'home economics', 'gardening',
                'agriculture', 'engineering technology', 'woodwork', 'metalwork',
                'electrical technology', 'automotive'
            ]
        }
        
        # Teaching and education skills
        teaching_skills = {
            'pedagogy': [
                'teaching', 'instruction', 'pedagogy', 'lesson planning',
                'curriculum development', 'assessment', 'evaluation'
            ],
            'classroom': [
                'classroom management', 'student engagement', 'behavior management',
                'differentiated instruction', 'inclusive education'
            ],
            'special_education': [
                'special needs', 'special education', 'learning disabilities',
                'individualized education program', 'adaptive learning'
            ]
        }
        
        found_skills = {
            'subjects': set(),
            'teaching': set(),
            'special_ed': set()
        }
        
        text_lower = text.lower()
        
        # Extract subject area skills
        for area, keywords in subject_areas.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_skills['subjects'].add(f"{area}: {keyword}")
        
        # Extract teaching skills
        for category, keywords in teaching_skills.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if category == 'special_education':
                        found_skills['special_ed'].add(keyword)
                    else:
                        found_skills['teaching'].add(keyword)
        
        # Combine all skills
        all_skills = list(found_skills['subjects']) + \
                    list(found_skills['teaching']) + \
                    list(found_skills['special_ed'])
        
        return all_skills
    
def process_resume(file_path):
    """Process resume with enhanced error handling"""
    try:
        # Verify file exists
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return None
            
        # Verify file is readable
        if not os.access(file_path, os.R_OK):
            st.error(f"File is not readable: {file_path}")
            return None
            
        # Create parser instance
        parser = CustomResumeParser(file_path)
        
        # Get data with timeout
        data = parser.get_extracted_data()
        
        # Validate returned data
        if not data:
            st.error("No data could be extracted from the resume")
            return None
            
        # Ensure all required fields are present
        required_fields = ['name', 'email', 'mobile_number', 'skills', 'no_of_pages']
        for field in required_fields:
            if field not in data or data[field] is None:
                data[field] = 'Not found' if field != 'skills' else []
                
        return data
        
    except Exception as e:
        st.error(f"Error processing resume: {str(e)}")
        st.error("Please make sure your PDF file is not corrupted and is properly formatted")
        return None

def check():
    """Check if the user is logged in and fetch user details."""
    if not st.session_state.get("logged_in"):
        st.warning("Please log in first.")
        return None
    
    if st.session_state.get("user_type", "").lower() != "tutor":
        st.error("Access denied. This page is for users only.")
        return None
    
    user_email = st.session_state.get("email")
    
    try:
        response = supabase.table('users').select('*').eq('email', user_email).execute()
        user = response.data
        if user:
            return user[0]  # Return the user object
        else:
            st.error("User details not found.")
    except Exception as e:
        st.error(f"Error fetching user details: {e}")

    return None

def display_user_dashboard(user):
    st.markdown(f"<h1 style='text-align: center; color: #021659;'>Welcome, {user['username']}!</h1>", unsafe_allow_html=True)
    st.write(f"Full Name: {user['full_name']}")
    st.write(f"Email: {user['email']}")

def extract_keywords_from_resume(resume_text):
    keywords = set()  # Use a set to avoid duplicates
    
    # Basic role patterns
    role_patterns = [
        r"\b(Teacher|Tutor|Instructor|Educator)\b",
        r"\b(Special Needs|Special Education|Special Children)\b",
        r"\b(Developer|Engineer|Manager|Designer|Analyst)\b"
    ]
    
    # Skill patterns
    skill_patterns = [
        r"\b(Teaching|Education|Learning|Instruction)\b",
        r"\b(Special Needs|Special Education|Adapted Learning)\b",
        r"\b(Python|Java|SQL|Communication|Leadership|PHP)\b",
        r"\b(Classroom Management|Student Assessment|Gamification)\b"
    ]
    
    # Extract all matches
    for pattern in role_patterns + skill_patterns:
        matches = re.findall(pattern, resume_text, re.IGNORECASE)
        keywords.update([match.lower() for match in matches if match])
    
    # Add explicit check for special education related terms
    special_ed_terms = ['special needs', 'special education', 'special children']
    for term in special_ed_terms:
        if term in resume_text.lower():
            keywords.add(term)
    
    return keywords

def show_pdf(file_path):
    with open(file_path, "rb") as f:
        pdf_data = f.read()
        b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="resume.pdf">Download PDF</a>'
        st.markdown(href, unsafe_allow_html=True)

def recommend_jobs_from_database(keywords):
    try:
        response = supabase.table('job_listings').select('*').execute()
        all_jobs = response.data
        
        recommended_jobs = []
        keywords = [k.lower().strip() for k in keywords]  # Normalize keywords
        
        # Add common variations of special education keywords
        special_ed_keywords = ['special needs', 'special education', 'special children']
        if any(keyword in special_ed_keywords for keyword in keywords):
            keywords.extend(special_ed_keywords)
        
        for job in all_jobs:
            matches = False
            
            # Check job description
            if 'job_description' in job:
                desc_lower = job['job_description'].lower()
                if any(keyword in desc_lower for keyword in keywords):
                    matches = True
            
            # Check job subject
            if 'job_subject' in job:
                subject_lower = job['job_subject'].lower()
                if any(keyword in subject_lower for keyword in keywords):
                    matches = True
            
            # Check required skills
            if 'required_skills' in job and job['required_skills']:
                skills_lower = job['required_skills'].lower()
                if any(keyword in skills_lower for keyword in keywords):
                    matches = True
                
                # Special check for special education keywords
                if any(term in skills_lower for term in special_ed_keywords):
                    if any(term in keywords for term in special_ed_keywords):
                        matches = True
            
            if matches:
                recommended_jobs.append(job)
                
        return recommended_jobs
    except Exception as e:
        st.error(f"Error querying database: {e}")
        return []

def get_table_download_link(df, filename, text):
    """Generates a link allowing the data in a given pandas dataframe to be downloaded."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def pdf_reader(file):
    """Enhanced PDF text extraction"""
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(file)
        if not text or text.isspace():
            return None
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def insert_data(name, email, res_score, timestamp, no_of_pages, reco_field, cand_level, skills, recommended_skills):
    """Insert user data into the database."""
    DB_table_name = 'user_data'
    rec_values = {
        "name": name,
        "email": email,
        "res_score": float(res_score),
        "timestamp": timestamp,
        "no_of_pages": int(no_of_pages),
        "reco_field": reco_field,
        "cand_level": cand_level,
        "skills": str(skills),
        "recommended_skills": str(recommended_skills)
    }

    try:
        supabase.table(DB_table_name).insert(rec_values).execute()
        print("Data inserted successfully!")
    except Exception as e:
        print(f"Error inserting data: {e}")

def run():
    user_id = check()  # Call check to verify login and get user_id
    if user_id is None:
        return

    display_user_dashboard(user_id)
    
    st.markdown('''
    <h2 style='text-align: center; color: #021659; border-bottom: 2px solid #021659; padding-bottom: 10px;'>Upload Your Resume for Smart Recommendations 💡</h2>
    ''', unsafe_allow_html=True)

    pdf_file = st.file_uploader("Choose your Resume (PDF)", type=["pdf"], label_visibility="collapsed")
    
    if pdf_file is not None:
        with st.spinner('Uploading your Resume...'):
            time.sleep(2)  # Reduced wait time for demonstration
        os.makedirs('./Uploaded_Resumes', exist_ok=True)
        save_image_path = './Uploaded_Resumes/' + pdf_file.name
        os.makedirs('./Uploaded_Resumes', exist_ok=True)
        with open(save_image_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        
        show_pdf(save_image_path)
        resume_data = process_resume(save_image_path)
        
        if resume_data:
            resume_text = pdf_reader(save_image_path)

            st.header("**Resume Analysis**")
            st.success(f"Hello **{resume_data['name']}**")
            st.subheader("**Your Basic Info**")
            st.write(f"**Name:** {resume_data['name']}")
            st.write(f"**Email:** {resume_data['email']}")
            st.write(f"**Contact:** {resume_data['mobile_number']}")
            st.write(f"**Resume pages:** {resume_data['no_of_pages']}")

            # Candidate Level Assessment
            cand_level = ''
            skill_count = len(resume_data.get('skills', []))
            if resume_data['no_of_pages'] == 1 and skill_count < 3:
                cand_level = "Fresher"
                st.markdown('<h4 style="color: #d73b5c; border: 1px solid #d73b5c; padding: 10px; border-radius: 5px;">You are at Fresher level!</h4>', unsafe_allow_html=True)
            elif resume_data['no_of_pages'] == 2 or (resume_data['no_of_pages'] == 1 and skill_count >= 3):
                cand_level = "Intermediate"
                st.markdown('<h4 style="color: #1ed760; border: 1px solid #1ed760; padding: 10px; border-radius: 5px;">You are at Intermediate level!</h4>', unsafe_allow_html=True)
            elif resume_data['no_of_pages'] >= 3:
                cand_level = "Experienced"
                st.markdown('<h4 style="color: #fba171; border: 1px solid #fba171; padding: 10px; border-radius: 5px;">You are at Experienced level!</h4>', unsafe_allow_html=True)

            # Skills input with tagging
            keywords = st_tags(label='### Your Current Skills', text='See our skills recommendations below', value=resume_data['skills'], key='1')
            
            # Normalizing skills
            keywords = [skill.lower().strip() for skill in resume_data['skills']]
            recommended_skills = []
            reco_field = ''
            
            for i in resume_data['skills']:
                if i.lower() in it_keywords:
                    reco_field = 'Information Technology'
                    st.success("**Our analysis suggests you are looking for Information Technology Jobs.**")
                    recommended_skills = ['Database Management', 'Digital Pedagogy', 'System Administration']
                    break
                elif i.lower() in software_keywords:
                    reco_field = 'Software Engineering'
                    st.success("**Our analysis suggests you are looking for Software Engineering Jobs.**")
                    recommended_skills = ['React', 'Usability Testing', 'Node JS']
                    break
                elif i.lower() in multimedia_keywords: 
                    reco_field = 'Multimedia'
                    st.success("**Our analysis suggests you are looking for Multimedia Jobs.**")
                    recommended_skills = ['Motion Graphics', 'Blender', 'User Interface Design']
                    break
                elif i.lower() in science_keywords:
                    reco_field = 'Science'
                    st.success("**Our analysis suggests you are looking for Science Jobs.**")
                    recommended_skills = ['Electrochemistry', 'Biodiversity', 'Science Experiments']
                    break
                elif i.lower() in math_keywords:
                    reco_field = 'Mathematics'
                    st.success("**Our analysis suggests you are looking for Mathematics Jobs.**")
                    recommended_skills = ['Visual Learning in Math', 'Hands-on Math Activities', 'Inquiry-based Learning in Math']
                    break

            # Job recommendations based on extracted keywords
            keywords = extract_keywords_from_resume(resume_text)
            job_recommendations = recommend_jobs_from_database(keywords)
            
            st.subheader("Job Recommendations 💼")
            if job_recommendations:
                for idx, job in enumerate(job_recommendations):
                    st.markdown(f"""
                    <div style="border: 1px solid #021659; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <strong>Job Title:</strong> {job['job_title']}<br>
                        <strong>Subject Area:</strong> {job['job_subject']}<br>
                        <strong>Description:</strong> {job['job_description']}<br>
                        <strong>Required Skills:</strong> {job['required_skills']}<br>
                        <strong>Hourly Rate:</strong> RM{job['hourly_rate']}/hour<br>
                    </div>
                    """, unsafe_allow_html=True)
                    # Use a unique key for each button by combining job title and index
                    if st.button(f"Apply for {job['job_title']}", key=f"apply_{job['job_title']}_{idx}"):
                        st.session_state['selected_job_title'] = job['job_title']
                        st.session_state['page'] = "apply"
                        st.rerun()
            else:
                st.warning("No matching jobs found based on your skills and qualifications.")

            # Time and date for the entry
            ts = time.time()
            cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            timestamp = str(cur_date + '_' + cur_time)

             # Resume writing recommendations
            st.subheader("**Resume Tips & Ideas💡**")
            resume_score = 0
            if 'Objective' in resume_text:
                resume_score = resume_score+20
                st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Objective</h4>''',unsafe_allow_html=True)
            else:
                st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add your career objective, it will give your career intension to the Recruiters.</h4>''',unsafe_allow_html=True)

            if 'Declaration'  in resume_text:
                resume_score = resume_score + 20
                st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Delcaration/h4>''',unsafe_allow_html=True)
            else:
                st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Declaration. It will give the assurance that everything written on your resume is true and fully acknowledged by you</h4>''',unsafe_allow_html=True)

            if 'Hobbies' or 'Interests'in resume_text:
                resume_score = resume_score + 20
                st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Hobbies</h4>''',unsafe_allow_html=True)
            else:
                st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Hobbies. It will show your persnality to the Recruiters and give the assurance that you are fit for this role or not.</h4>''',unsafe_allow_html=True)

            if 'Achievements' in resume_text:
                resume_score = resume_score + 20
                st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Achievements </h4>''',unsafe_allow_html=True)
            else:
                st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Achievements. It will show that you are capable for the required position.</h4>''',unsafe_allow_html=True)

            if 'Projects' in resume_text:
                resume_score = resume_score + 20
                st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects</h4>''',unsafe_allow_html=True)
            else:
                st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Projects. It will show that you have done work related the required position or not.</h4>''',unsafe_allow_html=True)

            st.subheader("**Resume Score📝**")
            st.markdown(
                """
                <style>
                    .stProgress > div > div > div > div {
                        background-color: #d73b5c;
                    }
                </style>""",
                unsafe_allow_html=True,
            )
            my_bar = st.progress(0)
            score = 0
            for percent_complete in range(resume_score):
                score +=1
                time.sleep(0.1)
                my_bar.progress(percent_complete + 1)
            st.success('** Your Resume Writing Score: ' + str(score)+'**')
            st.warning("** Note: This score is calculated based on the content that you have in your Resume. **")
            st.balloons()

            # Inserting data into the database
            insert_data(resume_data['name'], resume_data['email'], resume_score, timestamp,
                        resume_data['no_of_pages'], reco_field, cand_level, str(resume_data['skills']),
                        str(recommended_skills))

            # Bonus Videos
            st.header("**Bonus Video for Resume Writing Tips 💡**")
            resume_vid = random.choice(resume_videos)
            st.video(resume_vid)

            st.header("**Bonus Video for Interview Tips 💡**")
            interview_vid = random.choice(interview_videos)
            st.video(interview_vid)

        else:
            st.error('Something went wrong..')

if __name__ == "__main__":
    run()
