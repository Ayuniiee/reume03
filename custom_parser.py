import pdfplumber
import spacy
from spacy.language import Language
from pyresparser import ResumeParser
import os
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using pdfplumber"""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return None

def create_nlp():
    try:
        logger.info("Loading spaCy model...")
        nlp = spacy.load("en_core_web_sm")
        logger.info("SpaCy model loaded successfully")
        return nlp
    except Exception as e:
        logger.error(f"Error loading spaCy model: {e}")
        raise

def get_resume_data(file_path):
    try:
        logger.info(f"Starting resume parsing for file: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
            
        # First try to extract text
        text = extract_text_from_pdf(file_path)
        if not text or len(text.strip()) < 50:  # Check if we got meaningful text
            logger.error("Not enough text extracted from PDF")
            return None
            
        nlp = create_nlp()
        logger.info("Created NLP object")
        
        # Add basic spaCy pipeline components if they're missing
        if 'tagger' not in nlp.pipe_names:
            nlp.add_pipe('tagger')
        if 'parser' not in nlp.pipe_names:
            nlp.add_pipe('parser')
        if 'ner' not in nlp.pipe_names:
            nlp.add_pipe('ner')
            
        logger.info("Starting ResumeParser")
        parser = ResumeParser(file_path, nlp=nlp)
        logger.info("Getting extracted data")
        data = parser.get_extracted_data()
        
        # If we got text but no data, try manual extraction
        if not data or not data.get('name'):
            logger.info("Attempting manual data extraction")
            # Basic manual extraction
            data = {
                'name': '',  # You might want to add name extraction logic
                'email': extract_email(text),
                'mobile_number': extract_phone(text),
                'skills': extract_skills(text, nlp),
                'no_of_pages': len(text) // 3000 + 1  # Rough estimate
            }
            
        logger.info("Successfully extracted resume data")
        return data
        
    except Exception as e:
        logger.error(f"Error in get_resume_data: {e}", exc_info=True)
        return None

def extract_email(text):
    """Extract email from text using regex"""
    import re
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else ''

def extract_phone(text):
    """Extract phone number from text using regex"""
    import re
    phone_pattern = r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}'
    phones = re.findall(phone_pattern, text)
    return phones[0] if phones else ''

def extract_skills(text, nlp):
    """Extract skills from text using spaCy"""
    doc = nlp(text.lower())
    # Add your skill keywords here
    skill_keywords = {'python', 'java', 'c++', 'sql', 'javascript', 'teaching',
                     'communication', 'leadership', 'management', 'analysis'}
    skills = []
    for token in doc:
        if token.text in skill_keywords:
            skills.append(token.text)
    return list(set(skills))
 
