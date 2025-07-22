"""
Backend server for Windows platform - Blackbird SDK
"""
# Import configuration
from core.config import *

from routes.auth import auth_bp
from routes.training import *
from routes.chat import chat_bp
from routes.chat import *
from routes.payment import payment_bp
from routes.files import files_bp
from routes.image import *
from routes.confluence import confluence_bp
from routes.vision_chat import vision_chat_bp
# from blackbird_sdk.utils.model_downloader import ensure_models

# === Now import Flask and other modules ===
from flask import Flask, request, jsonify, Response, stream_with_context, render_template, redirect, send_from_directory, send_file
from flask_cors import CORS
from flask_session import Session
import subprocess
import logging
import threading
from datetime import datetime

# === Import local modules (these should now work) ===
try:
    from enums import ENCRYPTION_KEY_VALUE, STRIPE_API_KEY
    # print("✅ Successfully imported enums")
except ImportError as e:
    print(f"❌ Failed to import enums: {e}")
    # Create fallback values
  

try:
    import rag_chat
    # print("✅ Successfully imported rag_chat")
except ImportError as e:
    print(f"❌ Failed to import rag_chat: {e}")
    rag_chat = None

try:
    import global_state_setup
    # print("✅ Successfully imported global_state_setup")
except ImportError as e:
    print(f"❌ Failed to import global_state_setup: {e}")
    global_state_setup = None

try:
    import finetune_schedule
    # print("✅ Successfully imported finetune_schedule")
except ImportError as e:
    print(f"❌ Failed to import finetune_schedule: {e}")
    finetune_schedule = None

try:
    import content_filter
    # print("✅ Successfully imported content_filter")
except ImportError as e:
    print(f"❌ Failed to import content_filter: {e}")
    content_filter = None

try:
    import lora2
    # print("✅ Successfully imported lora2")
except ImportError as e:
    print(f"❌ Failed to import lora2: {e}")
    lora2 = None

try:
    import train_agent_initial
    # print("✅ Successfully imported train_agent_initial")
except ImportError as e:
    print(f"❌ Failed to import train_agent_initial: {e}")
    train_agent_initial = None

# === Continue with your existing imports ===
import json
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn

from flask import Flask, request, jsonify, Response, stream_with_context ,render_template,redirect, send_from_directory, send_file
from flask_cors import CORS
from flask_session import Session
import subprocess
import os
import json
from werkzeug.utils import secure_filename



# Add UTF-8 encoding fix to prevent 'charmap' codec errors with emoji characters
import sys
# Configure UTF-8 encoding for console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8')
# Set environment variable for Python I/O encoding
os.environ["PYTHONIOENCODING"] = "utf-8"
# On Windows, set console code page to UTF-8
if sys.platform == 'win32':
    try:
        # Use subprocess.run directly with shell=True for Windows
        subprocess.run('chcp 65001', check=False, shell=True)
        print("Console code page set to UTF-8 (65001)")
    except Exception as e:
        print(f"Warning: Could not set console code page: {e}")

from rag_chat import RAGChat , create_training_files_with_feedback , run_training_with_derived_paths
from pathlib import Path
import re
import os
from typing import Dict, List, Optional
# Set environment variables first
os.environ["OMP_NUM_THREADS"] = "4"  # Single env var for thread control
from typing import List, Dict
import sys
from datetime import datetime, timedelta
import sys
from huggingface_hub import snapshot_download, cached_assets_path, scan_cache_dir
from pydantic import BaseModel
from typing import List, Optional, Dict
import json
import os
from datetime import datetime
from global_state_setup import GlobalState
import gc
# ================
#  DATA MODELS
# ================
from finetune_schedule import AgentFineTuner
import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
# Enable PyTorch optimizations
torch.backends.cuda.matmul.allow_tf32 = True  # NVIDIA optimization
torch.set_float32_matmul_precision('high')    # CPU & Apple optimization
if torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (8,0):
    torch.backends.cuda.enable_flash_sdp(True)  # Enable Flash-Attention 2

# Additional imports for image generation
import uuid
from io import BytesIO
from PIL import Image, PngImagePlugin
import logging
import traceback

# Import the proper content filter and related functions
from content_filter import ContentFilter, add_watermark, add_metadata_to_image, dict_to_pnginfo

class UpdatePatternsRequest(BaseModel):
    agent_type: str
    input: str

class SuggestionsRequest(BaseModel):
    agent_type: str
    input: str
    max_suggestions: Optional[int] = 5

class TrainingDataItem(BaseModel):
    agent_type: str
    prompt: str

class InitialTrainingRequest(BaseModel):
    training_data: List[TrainingDataItem]

class AgentMetrics(BaseModel):
    total_interactions: int
    unique_patterns: int
    last_updated: str

class ModelState(BaseModel):
    agents: List[str]
    total_interactions: int
    metrics_by_agent: Dict[str, AgentMetrics]


from core.config import CORS_CONFIG, FLASK_CONFIG

app = Flask(__name__, static_folder='build', static_url_path='/static')
CORS(app, **CORS_CONFIG)
app.config.update(FLASK_CONFIG)
app.register_blueprint(auth_bp)
app.register_blueprint(chat_bp)
app.register_blueprint(files_bp)
app.register_blueprint(training_bp)
app.register_blueprint(payment_bp)
app.register_blueprint(image_bp)
app.register_blueprint(confluence_bp)
app.register_blueprint(vision_chat_bp)
print("✅ confluence_bp registered")

# Add request logging middleware
@app.before_request
def log_request():
    """Log all incoming requests"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n🔍 [{timestamp}] {request.method} {request.path}")
    print(f"   📍 Remote: {request.remote_addr}")
    print(f"   📋 Headers: {dict(request.headers)}")
    if request.is_json:
        print(f"   📄 JSON Body: {request.get_json()}")
    elif request.form:
        print(f"   📄 Form Data: {dict(request.form)}")
    print("   " + "="*50)

@app.after_request
def log_response(response):
    """Log all outgoing responses"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"📤 [{timestamp}] Response: {response.status_code} - {response.status}")
    if response.is_json:
        try:
            response_data = response.get_json()
            print(f"   📄 Response Body: {response_data}")
        except:
            print(f"   📄 Response Body: [JSON parsing failed]")
    print("   " + "="*50)
    return response

@app.after_request
def remove_hop_by_hop_headers(response):
    hop_by_hop = [
        'Connection',
        'Keep-Alive',
        'Proxy-Authenticate',
        'Proxy-Authorization',
        'TE',
        'Trailer',
        'Transfer-Encoding',
        'Upgrade'
    ]
    for header in hop_by_hop:
        response.headers.pop(header, None)
    return response

# Global variables
rag_chat = None
fine_tuning_active = False
flux_pipeline= False


def get_lifetime_storage():
    """Retrieve the lifetime storage used from a JSON file."""
    if not os.path.exists(STORAGE_TRACKER_FILE):
        return 0  # If no tracker file exists, assume 0 usage

    with open(STORAGE_TRACKER_FILE, "r") as f:
        try:
            data = json.load(f)
            return data.get("total_storage", 0)
        except json.JSONDecodeError:
            return 0  # If file is corrupted, reset usage

def update_lifetime_storage(new_size):
    """Update the total lifetime storage and persist it to a file."""
    total_storage = get_lifetime_storage() + new_size

    # Block further uploads if storage exceeds 2GB
    if total_storage > MAX_LIFETIME_STORAGE:
        return False  # Indicates that upload should be blocked

    # Save new storage usage
    with open(STORAGE_TRACKER_FILE, "w") as f:
        json.dump({"total_storage": total_storage}, f)

    return True  # Upload is allowed

@app.route('/api/storage-reset', methods=['POST'])
def reset_storage():
    """API to reset the lifetime storage counter (admin use only)."""
    try:
        with open(STORAGE_TRACKER_FILE, "w") as f:
            json.dump({"total_storage": 0}, f)

        return jsonify({'message': 'Lifetime storage counter reset to 0 bytes'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def setup_folders():
    # Create base directories for each agent and storage type
    agents = ['tech', 'legal', 'finance', 'meetings', 'general', 'research']
    storage_types = ['model_memory', 'saved_files']
    
    for agent in agents:
        for storage_type in storage_types:
            path = os.path.join(BASE_UPLOAD_FOLDER, agent, storage_type)
            os.makedirs(path, exist_ok=True)

setup_folders()
# Optimized text cleaning with pre-compiled regex patterns

BRACKETS_PATTERN = re.compile(r'\[.*?\]|\(.*?\)')
SPACES_PATTERN = re.compile(r'\s+')
CAMELCASE_PATTERN = re.compile(r'([a-z])([A-Z])')
NON_ASCII_PATTERN = re.compile(r'[^\x00-\x7F]+')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

import traceback

# @app.route('/format-python', methods=['POST'])
# def format_python():
#     # The raw code is sent as plain text in the request body
#     python_code = request.data.decode('utf-8')
    
#     # Use autopep8 to fix the code
#     formatted_code = autopep8.fix_code(python_code)
    
#     # Return the formatted code as plain text
#     return formatted_code, 200, {'Content-Type': 'text/plain'}
   
def clean_filename(filename):
    """Create a clean, readable version of the filename without special characters."""
    name = os.path.splitext(filename)[0]
    clean_name = ''.join(c if c.isalnum() else '_' for c in name)
    return clean_name
    
# Add this to test if your routes are accessible
@app.route('/debug-routes', methods=['GET'])
def debug_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'rule': rule.rule,
            'methods': list(rule.methods),
            'endpoint': rule.endpoint
        })
    return jsonify(routes)


def quantize(model: nn.Module, group_size: int, bits: int) -> nn.Module:
    """
    Applies quantization to the model weights using PyTorch quantization.

    Args:
        model (nn.Module): model to be quantized.
        group_size (int): group size for quantization.
        bits (int): bits per weight for quantization.

    Returns:
        nn.Module: The quantized model
    """
    # In PyTorch, we would use torch.quantization APIs
    # This is a simplified implementation - production code would use 
    # PyTorch's quantization workflow with calibration and proper observers

    # For dynamic quantization:
    if bits == 8:
        # Simple dynamic quantization for linear layers
        try:
            import torch.quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
            return quantized_model
        except Exception as e:
            print(f"Quantization failed: {e}")
            print("Returning original model instead")
            return model
    else:
        # For other bit configurations, we'd need more complex approaches
        # like custom quantizers or int4 quantization libraries
        print(f"PyTorch built-in quantization only supports 8-bit, requested {bits}-bit")
        print("Returning original model without quantization")
        return model


@app.route('/health', methods=['GET'])
def health():
    import os
    return jsonify({"status": "ok", "keepalive": os.environ.get("BLACKBIRD_KEEPALIVE") == "1"}), 200


# @app.errorhandler(Exception)
# def handle_error(e):
#     print("An error occurred:", str(e))
#     return jsonify({"error": str(e)}), 500


@app.route("/api/update-query", methods=["POST"])
def update_query():
    global global_state
    try:
        data = request.get_json()
        agent_type = data["agent_type"]
        input_text = data["input"]
        
        word_count = len(input_text.strip().split())

        # Enforce at least 3 words
        if word_count < 3:
            return jsonify({
                "error": "For the 'legal' agent, the query must contain at least 3 words."
            }), 400

        global_state.predictor.update_user_query(agent_type, input_text)
        global_state.save_if_needed()

        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/api/get-suggestions", methods=["POST"])
def get_suggestions_api():
    global global_state
    try:
        data = request.get_json()
        agent_type = data["agent_type"]
        partial_input = data["input"]
        max_suggestions = data.get("max_suggestions", 5)
        
        results = global_state.predictor.get_suggestions(
            agent_type=agent_type,
            partial_input=partial_input,
            max_suggestions=max_suggestions
        )
        return jsonify({"suggestions": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route("/api/save-model-state", methods=["POST"])
def save_model_state():
    global global_state
    try:
        global_state.save_state()
        return jsonify({"status": "success", "message": "Model state saved."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/load-model-state", methods=["POST"])
def load_model_state():
    global global_state
    try:
        global_state.load_state()
        return jsonify({"status": "success", "message": "Model state loaded."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


import pymysql
import requests
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests


# Global variables
training_lock = threading.Lock()
is_training = False

from waitress import serve
import logging

def start_background_tasks():
    """Start all background tasks when the app starts"""
    # Start the training monitor
    monitor_thread = threading.Thread(target=background_training_monitor)
    monitor_thread.daemon = True
    monitor_thread.start()
    print("Background training monitor started")

    

def main():
    import multiprocessing
    multiprocessing.freeze_support()
    # ensure_models() # This line is removed as per the edit hint
    logging.basicConfig(
        level=logging.INFO,               # Set the minimum logging level
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    print("\n" + "="*60)
    print("🚀 BLACKBIRD SDK SERVER STARTING")
    print("="*60)
    print(f"📍 Server URL: http://localhost:5012")
    print(f"🔍 Health Check: http://localhost:5012/health")
    print(f"📋 Debug Routes: http://localhost:5012/debug-routes")
    print("="*60)
    print("📝 Available Confluence Endpoints:")
    print("   POST /api/confluence/validate")
    print("   POST /api/confluence/test-connection")
    print("   POST /api/confluence/pages")
    print("   POST /api/confluence/page/<page_id>")
    print("="*60)
    print("🔄 Server is starting...\n")
    
    # Use Flask development server for better debugging
    app.run(host='0.0.0.0', port=5012, debug=True, threaded=True)
    
    # Alternative: Use waitress for production
    # serve(app, host='0.0.0.0', port=5012)

if __name__ == "__main__":
    main()
 
# Add this after your route definitions
print("=== FLASK ROUTES REGISTERED ===")
for rule in app.url_map.iter_rules():
    print(f"Route: {rule.rule} | Methods: {rule.methods} | Endpoint: {rule.endpoint}")
print("=== END ROUTES ===")

# from blackbird_sdk.utils.websearch_rag_pipeline import websearch_rag_pipeline

@app.route('/api/websearch-rag', methods=['POST'])
def websearch_rag():
    data = request.get_json()
    query = data.get('query')
    agent = data.get('agent')  # Not used yet, for future agent integration
    if not query:
        return jsonify({'error': 'Missing query'}), 400
    try:
        rag_knowledge = websearch_rag_pipeline(query)
        return jsonify({'rag_knowledge': rag_knowledge}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500