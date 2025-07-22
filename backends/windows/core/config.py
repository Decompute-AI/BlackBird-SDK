"""
Configuration module for Decompute Windows backend
Contains all global variables, constants, and configuration settings
"""

import os
from pathlib import Path
from datetime import timedelta

# === BACKEND DIRECTORY CONFIGURATION ===
BACKEND_DIR = Path(__file__).parent.parent.resolve()
# Define the GlobalState class
class GlobalState:
    def __init__(self):
        self.predictor = None
        self.is_training = False
        self.active_jobs = {}
        self.active_processes = {}
        self.training_lock = None
        self._modified = False
    
    def save_if_needed(self):
        if self._modified:
            print("Saving global state...")
            self._modified = False
            return True
        return False
    
    def mark_modified(self):
        self._modified = True
    
    def reset(self):
        self.predictor = None
        self.is_training = False
        self.active_jobs = {}
        self.active_processes = {}
        self._modified = False

# Create the global instance
global_state = GlobalState()

# === FALLBACK VALUES ===
ENCRYPTION_KEY_VALUE = "fallback_key"
STRIPE_API_KEY = "fallback_stripe_key"

# === FILE AND STORAGE CONFIGURATION ===
UPLOAD_FOLDER = os.path.expanduser('~/Documents/Decompute-Files/uploads')
BASE_UPLOAD_FOLDER = os.path.expanduser('~/Documents/Decompute-Files')
ALLOWED_EXTENSIONS = {'.txt', '.json', '.pdf', '.docx', '.wav', '.xlsx', 'xls', '.m4a', '.mp3', '.py', '.js'}

STORAGE_DIR = os.path.join(BASE_UPLOAD_FOLDER, 'storage')
MODEL_WEIGHTS_DIR = os.path.join(STORAGE_DIR, 'model_weights')
DOCUMENTS_DIR = os.path.join(STORAGE_DIR, 'documents')
MODEL_MEMORY_DIR = os.path.join(STORAGE_DIR, 'model_memory')

STORAGE_TRACKER_FILE = "storage_tracker.json"
MAX_LIFETIME_STORAGE = 100 * 1024 * 1024  # 100MB in bytes

# === DATABASE CONFIGURATION ===
DB_CONFIG = {
    'host': 'database-1.ch4ykucgglmt.us-east-2.rds.amazonaws.com',
    'user': 'admin',
    'password': 'Pikachu0619',
    'database': 'decompute',
    'port': 3306
}



# === ALLOWED COUNTRIES ===
ALLOWED_COUNTRIES = {'India', 'USA', 'United States'}

# === STRIPE CONFIGURATION ===
YOUR_DOMAIN = 'http://127.0.0.1:5012'

# === FLASK CONFIGURATION ===
FLASK_CONFIG = {
    'SECRET_KEY': os.environ.get('FLASK_SECRET_KEY', 'nvonrvonwkvnkenv'),
    'SESSION_COOKIE_SECURE': False,
    'SESSION_COOKIE_HTTPONLY': True,
    'SESSION_COOKIE_SAMESITE': 'None',
    'PERMANENT_SESSION_LIFETIME': timedelta(hours=1)
}

# === CORS CONFIGURATION ===
CORS_CONFIG = {
    'supports_credentials': True,
    'resources': {
        r"/*": {
            "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
            "methods": ["GET", "POST", "OPTIONS", "DELETE"],
            "allow_headers": ["Content-Type", "Accept"],
            "supports_credentials": True,
            "expose_headers": ["Set-Cookie"]
        }
    }
}

# === TRAINING CONFIGURATION ===
IDLE_THRESHOLD_SECONDS = 600  # 10 minutes
TRAINING_CHECK_INTERVAL = 300  # Check every 5 minutes
MIN_TRAINING_INTERVAL_HOURS = 24  # Wait 24 hours between training the same agent

# === FILE PATHS ===
training_history_file = os.path.expanduser('~/Library/Application Support/Decompute/training_history.json')
# === HUGGING FACE CONFIGURATION ===
HF_TOKEN = os.environ.get('HF_TOKEN', None)

# === TEXT PROCESSING CONFIGURATION ===
SPECIAL_CHARS = {
    '\u2013': '-', '\u2014': '-', '\u2018': "'", '\u2019': "'", '\u201c': '"',
    '\u201d': '"', '\u2022': '*', '\u2026': '...', '\u2192': '->', '\u25a0': '',
    '\u00f6': 'o', '\u00e9': 'e', '\u00e1': 'a', '\u00ed': 'i', '\u00f3': 'o',
    '\u00fa': 'u', '\u00f1': 'n', '\u00df': 'ss', '\u2264': '<=', '\u2265': '>=',
    '\u00b0': ' degrees ', '\u00b5': 'u', '\u00b1': '+/-', '\u03b1': 'alpha',
    '\u03b2': 'beta', '\u03b3': 'gamma', '\u03bc': 'mu'
}

# === ENSURE DIRECTORIES EXIST ===
def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        UPLOAD_FOLDER, BASE_UPLOAD_FOLDER, STORAGE_DIR, 
        MODEL_WEIGHTS_DIR, DOCUMENTS_DIR, MODEL_MEMORY_DIR
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Create directories on import
ensure_directories()
