import fitz  # PyMuPDF
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)


def extract_text_from_pdf(uploaded_file):
    """
    Extracts text from a PDF file.
    
    Args:
        uploaded_file: A file-like object for the PDF (e.g., from a web form).
        
    Returns:
        str: The extracted text.
    """
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def ask_gemini(prompt, model_name="models/gemini-1.5-flash", max_tokens=500):
    """
    Sends a prompt to the Gemini API (Gemini 1.5 Flash) and returns the response.
    
    Args:
        prompt (str): The text prompt to send to Gemini.
        model_name (str): Gemini model name (default: "models/gemini-1.5-flash").
        max_tokens (int): Max response tokens.
        
    Returns:
        str: Gemini's response.
    """
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text
