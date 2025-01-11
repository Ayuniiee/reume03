from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

class JobMatcher:
    def __init__(self, supabase_client=None):
        """
        Initialize JobMatcher with an optional Supabase client
        If no client is provided, it will run without database features
        """
        self.supabase = supabase_client
        self.vectorizer = TfidfVectorizer(stop_words='english')
        # Initialize an in-memory field knowledge map for testing/backup
        self.field_knowledge_map = {
            "computer science": {
                "related_fields": "software development\ndata science\nweb development\ncloud computing",
                "knowledge_areas": "programming\nalgorithms\ndata structures\nsoftware engineering"
            },
            "data science": {
                "related_fields": "machine learning\ndata analysis\nstatistics\nbusiness intelligence",
                "knowledge_areas": "statistics\nmachine learning\ndata analysis\npython\nSQL"
            }
            # Add more fields as needed
        }

    def get_field_knowledge(self, degree_field):
        """Get field knowledge either from Supabase or fallback to local data"""
        try:
            if self.supabase:
                result = self.supabase.table('field_knowledge_map')\
                    .select('*')\
                    .eq('degree_field', degree_field)\
                    .execute()
                if result.data:
                    return result.data[0]
            
            # Fallback to local data if no Supabase or no results
            return self.field_knowledge_map.get(degree_field.lower(), {
                "related_fields": "",
                "knowledge_areas": ""
            })
        except Exception as e:
            print(f"Error fetching field knowledge: {e}")
            return {"related_fields": "", "knowledge_areas": ""}

    def get_matches(self, resume_data, job_listings):
        """Find matching jobs for a given resume"""
        if not isinstance(resume_data, dict) or not isinstance(job_listings, list):
            raise ValueError("resume_data must be a dictionary and job_listings must be a list")

        required_resume_fields = ['degree_field', 'skills']
        if not all(field in resume_data for field in required_resume_fields):
            raise ValueError(f"resume_data must contain all required fields: {required_resume_fields}")

        matches = []
        for job in job_listings:
            required_job_fields = ['id', 'job_title', 'job_description', 'required_skills']
            if not all(field in job for field in required_job_fields):
                print(f"Skipping job due to missing required fields: {job.get('job_title', 'Unknown')}")
                continue

            score = self.calculate_match_score(resume_data, job)
            if score >= 0.3:  # Minimum matching threshold
                matches.append({
                    'job_id': job['id'],
                    'job_title': job['job_title'],
                    'match_score': round(score, 2),
                    'reason': self.get_match_reason(resume_data, job)
                })
        
        return sorted(matches, key=lambda x: x['match_score'], reverse=True)
    
    def calculate_match_score(self, resume_data, job):
        """Calculate similarity between resume and job"""
        try:
            field_data = self.get_field_knowledge(resume_data['degree_field'])
            
            if not field_data:
                return 0.0
            
            # Calculate different aspects of matching
            field_match = self._calculate_field_match(
                field_data.get('related_fields', ''),
                job['job_description']
            )
            
            skill_match = self._calculate_similarity(
                resume_data.get('skills', ''),
                job.get('required_skills', '')
            )
            
            knowledge_match = self._calculate_similarity(
                field_data.get('knowledge_areas', ''),
                job['job_description']
            )
            
            return (0.3 * field_match +
                    0.4 * skill_match +
                    0.3 * knowledge_match)
                    
        except Exception as e:
            print(f"Error calculating match score: {e}")
            return 0.0
    
    def _calculate_similarity(self, text1, text2):
        """Calculate TF-IDF similarity between two texts"""
        if not isinstance(text1, str) or not isinstance(text2, str):
            text1 = str(text1) if text1 else ''
            text2 = str(text2) if text2 else ''
            
        if not text1.strip() or not text2.strip():
            return 0.0
            
        try:
            matrix = self.vectorizer.fit_transform([text1, text2])
            return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def _calculate_field_match(self, related_fields, job_description):
        """Check if job matches any related fields"""
        if not isinstance(related_fields, str) or not isinstance(job_description, str):
            return 0.0
            
        if not related_fields.strip() or not job_description.strip():
            return 0.0
            
        fields = related_fields.lower().split('\n')
        return float(any(field in job_description.lower() for field in fields))

    def get_match_reason(self, resume_data, job):
        """Generate explanation for why this job matches"""
        reasons = []
        
        try:
            field_data = self.get_field_knowledge(resume_data['degree_field'])
            
            if field_data and field_data.get('related_fields'):
                fields = field_data['related_fields'].lower().split('\n')
                matched_fields = [
                    field for field in fields
                    if field in job['job_description'].lower()
                ]
                if matched_fields:
                    reasons.append(f"Your {resume_data['degree_field']} background aligns with {', '.join(matched_fields)}")
            
            if resume_data.get('skills') and job.get('required_skills'):
                skill_score = self._calculate_similarity(resume_data['skills'], job['required_skills'])
                if skill_score > 0.3:
                    reasons.append("Your skills match the job requirements")
            
            return " | ".join(reasons) if reasons else "General field alignment"
            
        except Exception as e:
            print(f"Error generating match reason: {e}")
            return "Unable to generate detailed match reason"