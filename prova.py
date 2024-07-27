from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import whisper
import os
import nltk
from nltk.tokenize import sent_tokenize
import requests
import logging
from dotenv import load_dotenv
import re
import spacy
import json
from datetime import datetime

# Load environment variables from a .env file
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download necessary data for nltk
nltk.download('punkt')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the Whisper model
model = whisper.load_model("base")

# Pexels API key from environment variable
PEXELS_API_KEY = os.getenv('PEXELS_API_KEY')
if not PEXELS_API_KEY:
    raise ValueError("PEXELS_API_KEY not found in environment variables.")

PEXELS_API_URL = 'https://api.pexels.com/videos/search'

# OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

OPENAI_API_URL = 'https://api.openai.com/v1/completions'

# Function to clean and tokenize text for more accurate video search
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

# Function to extract main keywords from a text using spaCy
def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in ('NOUN', 'PROPN') and not token.is_stop]
    return keywords[:5]  # Limit to the top 5 keywords

# Function to generate search queries based on keywords
def generate_queries(keywords):
    queries = []
    for keyword in keywords:
        queries.append(f"{keyword}")
        queries.append(f"car {keyword}")
        queries.append(f"{keyword} market")
        queries.append(f"man buying a car or a {keyword}")
    return queries

# Function to interact with OpenAI API to generate queries based on text input
def generate_queries_from_text(text):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {OPENAI_API_KEY}',
    }
    data = {
        'model': 'text-davinci-003',
        'max_tokens': 50,
        'prompt': f"Given the text: \"{text}\", generate relevant search queries.",
        'temperature': 0.7,
        'stop': '\n',
    }
    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        queries = [item['text'].strip() for item in result['choices'][0]['text'].strip().split('\n')]
        return queries
    except requests.RequestException as e:
        logging.error(f"Error generating queries from OpenAI: {e}")
    return []

# Function to search for videos on Pexels
def search_videos(query, count=5):
    headers = {
        'Authorization': PEXELS_API_KEY
    }
    params = {
        'query': query,
        'per_page': count
    }
    try:
        response = requests.get(PEXELS_API_URL, headers=headers, params=params)
        response.raise_for_status()
        logging.info(f"Searching Pexels for query: {query}")
        data = response.json()
        video_urls = []
        for video in data.get('videos', [])[:count]:  # Limit to `count` videos per query
            video_urls.append(video['video_files'][0]['link'])
        return video_urls
    except requests.RequestException as e:
        logging.error(f"Error while searching Pexels: {e}")
    return []

# Constants for word count constraints
MIN_WORDS_PER_PARAGRAPH = 10
MAX_WORDS_PER_PARAGRAPH = 50

@app.route('/')
def home():
    recent_projects = load_recent_projects()
    return render_template('home.html', recent_projects=recent_projects)

@app.route('/new_project', methods=['POST'])
def new_project():
    project_name = request.form['project_name']
    project_id = create_project_directory(project_name)
    session['project_id'] = project_id
    session['project_name'] = project_name  # Aggiungi questa linea per salvare il nome del progetto nella sessione
    return redirect(url_for('index'))

def load_project(project_id):
    project_path = os.path.join('projects', project_id)
    if not os.path.exists(project_path):
        return None

    data_path = os.path.join(project_path, 'data.json')
    if not os.path.exists(data_path):
        return None

    with open(data_path, 'r') as f:
        project_data = json.load(f)
    
    return project_data

@app.route('/open_project/<project_id>', methods=['GET'])
def open_project(project_id):
    project_data = load_project(project_id)
    if project_data is None:
        return redirect(url_for('index'))

    return render_template('project_details.html', project_id=project_id, project_data=project_data)

@app.route('/project/<project_id>')
def project_details(project_id):
    project_data = load_project(project_id)
    if project_data is None:
        return "Project not found", 404

    return render_template('project_details.html', project_id=project_id, project_data=project_data)


@app.route('/index')
def index():
    if 'project_id' not in session:
        return redirect(url_for('home'))
    project_name = session.get('project_name', 'Dashboard')  # Recupera il nome del progetto dalla sessione
    return render_template('index.html', project_name=project_name)

##
@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'project_id' not in session:
        return jsonify({"error": "Project not found"}), 400

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        try:
            project_id = session['project_id']
            project_path = os.path.join('projects', project_id)
            if not os.path.exists(project_path):
                return jsonify({"error": "Project directory not found"}), 400

            file_path = os.path.join(project_path, file.filename)
            file.save(file_path)

            # Support for additional audio formats
            valid_extensions = ['.mp3', '.wav', '.flac', '.m4a']
            _, file_extension = os.path.splitext(file.filename)
            if file_extension not in valid_extensions:
                return jsonify({"error": f"Unsupported file format: {file_extension}"}), 400

            result = model.transcribe(file_path, fp16=False)
            os.remove(file_path)  # Clean up the saved file

            # Split the transcribed text into paragraphs based on word count
            transcription_text = result["text"]
            paragraphs = split_text_into_paragraphs(transcription_text)

            # Generate queries and search videos for each paragraph
            paragraph_videos = []
            for paragraph in paragraphs:
                cleaned_paragraph = clean_text(paragraph)
                keywords = extract_keywords(cleaned_paragraph)
                if keywords:
                    queries = generate_queries(keywords)
                    videos = []
                    for query in queries:
                        videos.extend(search_videos(query, count=5))  # Fetch up to 5 videos per query
                        if len(videos) >= 5:
                            break  # Ensure only up to 5 videos per paragraph
                    paragraph_videos.append({
                        'paragraph': paragraph,
                        'keywords': keywords,
                        'videos': videos[:5]  # Limit to 5 videos per paragraph
                    })
                else:
                    paragraph_videos.append({
                        'paragraph': paragraph,
                        'keywords': [],
                        'videos': []
                    })
                logging.info(f"Paragraph: {paragraph}")
                logging.info(f"Keywords: {keywords}")
                logging.info(f"Videos: {videos}")

            save_project_data(project_id, file.filename, paragraph_videos)

            return jsonify({"paragraph_videos": paragraph_videos})
        except Exception as e:
            logging.error(f"Error processing the file: {e}")
            return jsonify({"error": "Error processing the file"}), 500

##


def split_text_into_paragraphs(text):
    paragraphs = []
    current_paragraph = []
    word_count = 0

    sentences = sent_tokenize(text)
    for sentence in sentences:
        current_paragraph.append(sentence)
        word_count += len(sentence.split())
        
        # Check if the paragraph reaches the maximum word limit
        if word_count >= MAX_WORDS_PER_PARAGRAPH:
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []
            word_count = 0
        # Check if the paragraph reaches the minimum word limit and there is a full stop
        elif word_count >= MIN_WORDS_PER_PARAGRAPH and sentence.endswith('.'):
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []
            word_count = 0
    
    # Add the remaining sentences as the last paragraph
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    return paragraphs

def create_project_directory(project_name):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    project_id = f"{project_name}"
    project_path = os.path.join('projects', project_id)
    os.makedirs(project_path, exist_ok=True)
    return project_id

def save_project_data(project_id, audio_filename, paragraph_videos):
    project_path = os.path.join('projects', project_id)
    if not os.path.exists(project_path):
        os.makedirs(project_path)

    data_path = os.path.join(project_path, 'data.json')
    project_data = {
        'audio_filename': audio_filename,
        'paragraph_videos': paragraph_videos
    }

    with open(data_path, 'w') as f:
        json.dump(project_data, f)

def load_recent_projects():
    if not os.path.exists('projects'):
        return []

    projects = []
    for project_id in os.listdir('projects'):
        project_path = os.path.join('projects', project_id)
        if os.path.isdir(project_path):
            data_path = os.path.join(project_path, 'data.json')
            if os.path.exists(data_path):
                with open(data_path, 'r') as f:
                    project_data = json.load(f)
                    projects.append({
                        'project_id': project_id,
                        'audio_filename': project_data['audio_filename'],
                        'paragraph_videos': project_data['paragraph_videos']
                    })
    return projects

if __name__ == "__main__":
    app.run(debug=True)
