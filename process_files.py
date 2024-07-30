import json
from youtube_dataset_prep import main as process_youtube
# Import PDF processing library (e.g., PyPDF2 or pdfplumber)
import PyPDF2
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def process_youtube_url(url):
    from youtube_dataset_prep import main as process_youtube, video_id_from_url
    
    video_id = video_id_from_url(url)
    process_youtube(url)  # This will create the full transcript file
    
    # Read the full transcript
    with open(f"{video_id}_full_transcript.txt", "r", encoding="utf-8") as f:
        full_transcript = f.read()
    
    # Clean and segment the full transcript
    from youtube_dataset_prep import clean_text, segment_text
    cleaned_text = clean_text(full_transcript)
    chunks = segment_text(cleaned_text)
    
    return chunks

def process_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    # Use the existing functions from youtube_dataset_prep.py to clean and segment the text
    from youtube_dataset_prep import clean_text, segment_text
    cleaned_text = clean_text(text)
    chunks = segment_text(cleaned_text)
    return chunks

def process_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    # Assuming the JSON file contains a list of text chunks
    return data

def prepare_dataset(file_type, file_path):
    logger.debug(f"Preparing dataset for file type: {file_type}")
    try:
        if file_type == 'youtube':
            result = process_youtube_url(file_path)
            logger.debug(f"YouTube processing result: {result}")
            return result
        elif file_type == 'pdf':
            return process_pdf(file_path)
        elif file_type == 'json':
            return process_json(file_path)
        else:
            raise ValueError("Unsupported file type")
    except Exception as e:
        logger.exception(f"Error in prepare_dataset: {str(e)}")
        return []  # Return an empty list instead of None