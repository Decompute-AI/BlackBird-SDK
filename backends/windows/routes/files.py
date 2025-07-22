"""
File management routes module for Decompute Windows backend
Handles file uploads, downloads, deletion, and model management
"""
from routes.chat import get_conversation_history
from huggingface_hub import scan_cache_dir
from flask import Blueprint, request, jsonify, send_file
import os
import json
import shutil
import tempfile
import threading
import time
from datetime import datetime
from werkzeug.utils import secure_filename
from pathlib import Path

# Import from core modules
from core.config import (
    BASE_UPLOAD_FOLDER, ALLOWED_EXTENSIONS, UPLOAD_FOLDER,
    MAX_LIFETIME_STORAGE, STORAGE_TRACKER_FILE, MODEL_WEIGHTS_DIR,
    DOCUMENTS_DIR, MODEL_MEMORY_DIR, STORAGE_DIR
)

# Create blueprint
files_bp = Blueprint('files', __name__)

def get_agent_folder(agent_id, storage_type):
    """Get the appropriate folder path for an agent and storage type"""
    return os.path.join(BASE_UPLOAD_FOLDER, agent_id, storage_type)

def clean_filename(filename: str) -> str:
    """Simple clean-up function for filenames"""
    return "".join(c for c in filename if c.isalnum() or c in (' ', '_', '-')).rstrip()

def process_file_info(file_path, storage_type):
    """Extract and format file information"""
    try:
        stat = os.stat(file_path)
        return {
            'name': os.path.basename(file_path),
            'size': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'storage_type': storage_type,
            'path': file_path
        }
    except Exception as e:
        print(f"Error processing file info for {file_path}: {str(e)}")
        return None

def update_storage_tracking(agent_id, file_size, operation='add'):
    """Update storage tracking for an agent"""
    try:
        storage_file = os.path.join(BASE_UPLOAD_FOLDER, agent_id, STORAGE_TRACKER_FILE)
        
        # Load existing data
        if os.path.exists(storage_file):
            with open(storage_file, 'r') as f:
                data = json.load(f)
        else:
            data = {'total_storage': 0, 'files': []}
        
        # Update total storage
        if operation == 'add':
            data['total_storage'] += file_size
        elif operation == 'remove':
            data['total_storage'] = max(0, data['total_storage'] - file_size)
        
        # Save updated data
        os.makedirs(os.path.dirname(storage_file), exist_ok=True)
        with open(storage_file, 'w') as f:
            json.dump(data, f)
            
        return data['total_storage']
        
    except Exception as e:
        print(f"Error updating storage tracking: {str(e)}")
        return 0
@files_bp.route('/api/storage-usage', methods=['GET'])
def get_storage_usage(agent_id):
    """Get current storage usage for an agent"""
    try:
        storage_file = os.path.join(BASE_UPLOAD_FOLDER, agent_id, STORAGE_TRACKER_FILE)
        if os.path.exists(storage_file):
            with open(storage_file, 'r') as f:
                data = json.load(f)
                return jsonify({'total_used': data.get('total_storage', 0), 'limit': MAX_LIFETIME_STORAGE }), 200
        
    except Exception as e:
        return jsonify(f"Error getting storage usage: {str(e)}"),500
        
# @app.route('/api/storage-usage', methods=['GET'])
# def get_storage_usage():
#     """API to get the total lifetime storage used."""
#     try:
#         total_size = get_lifetime_storage()
#         return jsonify({'total_used': total_size, 'limit': MAX_LIFETIME_STORAGE}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
@files_bp.route('/api/files/<agent_id>', methods=['GET'])
def get_processed_files(agent_id):
    try:
        model_memory_path = get_agent_folder(agent_id, 'model_memory')
        saved_files_path = get_agent_folder(agent_id, 'saved_files')
        files = {
            'modelMemory': [],
            'savedFiles': []
        }

        # Get directories from model memory
        if os.path.exists(model_memory_path):
            for dir_name in os.listdir(model_memory_path):
                if dir_name == '.DS_Store':
                    continue

                dir_path = os.path.join(model_memory_path, dir_name)
                
                # If it's not a directory, skip it
                if not os.path.isdir(dir_path):
                    continue

                # Check for model files (.npz or .safetensors)
                has_model_files = any(f.endswith(('.npz', '.safetensors', '.faiss')) for f in os.listdir(dir_path))
                if not has_model_files:
                    continue

                # If it passes the check, process the directory info
                dir_info = process_file_info(dir_path, 'model_memory')

                # Add display name from config.json if it exists
                config_path = os.path.join(dir_path, 'config.json')
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            dir_info['display_name'] = config.get('display_name', '')
                    except:
                        dir_info['display_name'] = ''
                else:
                    dir_info['display_name'] = ''

                # Add conversation history
                dir_info['conversations'] = get_conversation_history(dir_path)
                files['modelMemory'].append(dir_info)

        # Get directories and files from saved files
        if os.path.exists(saved_files_path):
            for entry_name in os.listdir(saved_files_path):
                if entry_name == '.DS_Store':
                    continue

                entry_path = os.path.join(saved_files_path, entry_name)

                # Handle both files and directories
                if os.path.isdir(entry_path):
                    # For directories, check for model files only if it's in model_memory
                    if 'model_memory' in entry_path:
                        has_model_files = any(f.endswith(('.npz', '.safetensors', '.faiss')) for f in os.listdir(entry_path))
                        if not has_model_files:
                            continue

                    # Process directory info
                    dir_info = process_file_info(entry_path, 'saved_files')

                    # Add display name from config.json if it exists
                    config_path = os.path.join(entry_path, 'config.json')
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                                dir_info['display_name'] = config.get('display_name', '')
                        except:
                            dir_info['display_name'] = ''
                    else:
                        dir_info['display_name'] = ''

                    # Add conversation history
                    dir_info['conversations'] = get_conversation_history(entry_path)
                    files['savedFiles'].append(dir_info)
                else:
                    # It's a single file - add it directly
                    file_info = process_file_info(entry_path, 'saved_files')
                    file_info['display_name'] = os.path.splitext(entry_name)[0]  # Use filename without extension
                    file_info['conversations'] = []  # Empty conversations for single files
                    files['savedFiles'].append(file_info)

        # Sort directories by date
        files['modelMemory'].sort(key=lambda x: x['date'], reverse=True)
        files['savedFiles'].sort(key=lambda x: x['date'], reverse=True)
        return jsonify(files)
    except Exception as e:
        print(f"Error getting files: {str(e)}")
        return jsonify({'error': str(e)}), 500

@files_bp.route('/api/files/<agent_id>', methods=['POST'])
def save_files(agent_id):
    """Upload and save files for a specific agent"""
    try:
        # Check if files are in the request
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        storage_type = request.form.get('storage_type', 'saved_files')
        
        # Validate storage type
        if storage_type not in ['model_memory', 'saved_files']:
            return jsonify({'error': 'Invalid storage type'}), 400
        
        # Check current storage usage
        current_storage = get_storage_usage(agent_id)
        
        # Calculate total size of new files
        total_new_size = 0
        temp_files = []
        
        for file in files:
            if file.filename == '':
                continue
            
            # Save to temporary location first
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            file.save(temp_file.name)
            temp_file.close()
            
            file_size = os.path.getsize(temp_file.name)
            total_new_size += file_size
            
            temp_files.append({
                'temp_path': temp_file.name,
                'original_name': file.filename,
                'size': file_size
            })
        
        # Check storage limit
        if current_storage + total_new_size > MAX_LIFETIME_STORAGE:
            # Clean up temp files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file['temp_path'])
                except:
                    pass
            
            return jsonify({
                'error': 'Storage limit exceeded',
                'current_storage': current_storage,
                'limit': MAX_LIFETIME_STORAGE,
                'attempted_size': total_new_size
            }), 400
        
        # Create agent folder
        folder_path = get_agent_folder(agent_id, storage_type)
        os.makedirs(folder_path, exist_ok=True)
        
        uploaded_files = []
        
        # Move files from temp to final location
        for temp_file_info in temp_files:
            try:
                original_filename = temp_file_info['original_name']
                file_extension = os.path.splitext(original_filename)[1].lower()
                
                # Check allowed extensions
                if file_extension not in ALLOWED_EXTENSIONS:
                    os.unlink(temp_file_info['temp_path'])
                    continue
                
                # Generate safe filename
                base_name = clean_filename(os.path.splitext(original_filename)[0])
                safe_filename = f"{base_name}{file_extension}"
                
                # Handle filename conflicts
                counter = 1
                final_path = os.path.join(folder_path, safe_filename)
                
                while os.path.exists(final_path):
                    name_part = f"{base_name}_{counter}"
                    safe_filename = f"{name_part}{file_extension}"
                    final_path = os.path.join(folder_path, safe_filename)
                    counter += 1
                
                # Move file to final location
                shutil.move(temp_file_info['temp_path'], final_path)
                
                # Update storage tracking
                update_storage_tracking(agent_id, temp_file_info['size'], 'add')
                
                # Add to uploaded files list
                file_info = process_file_info(final_path, storage_type)
                if file_info:
                    uploaded_files.append(file_info)
                
            except Exception as e:
                print(f"Error processing file {temp_file_info['original_name']}: {str(e)}")
                # Clean up temp file if it still exists
                try:
                    os.unlink(temp_file_info['temp_path'])
                except:
                    pass
        
        return jsonify({
            'message': f'Successfully uploaded {len(uploaded_files)} files',
            'files': uploaded_files,
            'storage_usage': get_storage_usage(agent_id)
        })
        
    except Exception as e:
        print(f"Error saving files: {str(e)}")
        return jsonify({'error': str(e)}), 500

@files_bp.route('/api/files/<agent_id>', methods=['DELETE'])
def delete_files(agent_id):
    try:
        data = request.json
        files_to_delete = data.get('files', [])

        # Iterate through each file-to-delete instruction
        for file_info in files_to_delete:
            name = file_info.get('name')
            storage_type = file_info.get('storage_type')

            if not name or not storage_type:
                continue  # Skip invalid entries

            # Determine the correct folder path based on storage_type
            if storage_type == 'model_memory':
                folder_path = get_agent_folder(agent_id, 'model_memory')
            elif storage_type == 'saved_files':
                folder_path = get_agent_folder(agent_id, 'saved_files')
            else:
                continue  # Skip unknown storage_type

            # Full path of the item to be deleted
            item_path = os.path.join(folder_path, name)

            # Check if item exists
            if os.path.exists(item_path):
                try:
                    # If it's a directory, remove it recursively
                    if os.path.isdir(item_path):
                        # On Windows, try multiple times to handle file locks
                        max_retries = 3
                        retry_delay = 1  # seconds
                        for attempt in range(max_retries):
                            try:
                                shutil.rmtree(item_path, ignore_errors=True)
                                break
                            except Exception as e:
                                if attempt == max_retries - 1:
                                    raise
                                time.sleep(retry_delay)
                    else:
                        # If it's a single file, remove the file
                        # On Windows, try multiple times to handle file locks
                        max_retries = 3
                        retry_delay = 1  # seconds
                        for attempt in range(max_retries):
                            try:
                                os.remove(item_path)
                                break
                            except Exception as e:
                                if attempt == max_retries - 1:
                                    raise
                                time.sleep(retry_delay)
                except Exception as e:
                    print(f"Error deleting {item_path}: {str(e)}")
                    return jsonify({'error': f'Failed to delete {name}: {str(e)}'}), 500

        return jsonify({'message': 'Selected file(s) deleted successfully.'}), 200

    except Exception as e:
        print(f"Error deleting files: {str(e)}")
        return jsonify({'error': str(e)}), 500

@files_bp.route('/api/check-filename-exists', methods=['POST'])
def check_filename_exists():
    try:
        data = request.json
        agent = data.get('agent')
        filename = data.get('filename')
        
        if not agent or not filename:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Define the path to check based on your file structure
        model_memory_path = get_agent_folder(agent, 'model_memory')
        saved_files_path = get_agent_folder(agent, 'saved_files')
        
        # Check if a file with this name exists in either storage location
        exists = False
        
        # Check model memory files
        if os.path.exists(model_memory_path):
            for dir_name in os.listdir(model_memory_path):
                if dir_name == '.DS_Store':
                    continue
                dir_path = os.path.join(model_memory_path, dir_name)
                if os.path.isdir(dir_path):
                    config_path = os.path.join(dir_path, 'config.json')
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                                if config.get('display_name') == filename:
                                    exists = True
                                    break
                        except:
                            pass
        
        # Check saved files if not found in model memory
        if not exists and os.path.exists(saved_files_path):
            for dir_name in os.listdir(saved_files_path):
                if dir_name == '.DS_Store':
                    continue
                dir_path = os.path.join(saved_files_path, dir_name)
                if os.path.isdir(dir_path):
                    config_path = os.path.join(dir_path, 'config.json')
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                                if config.get('display_name') == filename:
                                    exists = True
                                    break
                        except:
                            pass
        
        return jsonify({'exists': exists})
    except Exception as e:
        print(f"Error checking filename: {str(e)}")
        return jsonify({'error': str(e), 'exists': False}), 500

@files_bp.route('/downloaded-models', methods=['GET'])
def get_downloaded_models():
    try:
        # Get Hugging Face cache directory
        cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
        
        # Get cache info
        hf_cache_info = scan_cache_dir()
        
        # Create a list of model info
        models_info = []
        
        # Iterate through repositories in the cache
        for repo in hf_cache_info.repos:
            # Only process model repositories
            if repo.repo_type == 'model':
                try:
                    # Handle last_modified timestamp conversion
                    last_modified = None
                    if repo.last_modified is not None:
                        if isinstance(repo.last_modified, (int, float)):
                            # Convert timestamp to datetime
                            last_modified = datetime.fromtimestamp(repo.last_modified).isoformat()
                        else:
                            last_modified = repo.last_modified.isoformat()
                    
                    # Read model info from config if available
                    config_path = os.path.join(repo.repo_path, 'config.json')
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                        models_info.append({
                            'name': repo.repo_id,
                            'path': str(repo.repo_path),
                            'config': config,
                            'size': repo.size_on_disk,
                            'last_modified': last_modified
                        })
                    else:
                        models_info.append({
                            'name': repo.repo_id,
                            'path': str(repo.repo_path),
                            'size': repo.size_on_disk,
                            'last_modified': last_modified
                        })
                except Exception as e:
                    print(f"Error processing repo {repo.repo_id}: {str(e)}")
                    # Add the model even if there's an error, without the problematic fields
                    models_info.append({
                        'name': repo.repo_id,
                        'path': str(repo.repo_path),
                        'size': repo.size_on_disk
                    })
        
        return jsonify(models_info)
    
    except Exception as e:
        print(f"Error in get_downloaded_models: {str(e)}")
        return jsonify({'error': str(e)}), 500


def get_directory_size(directory):
    """Calculate total size of a directory"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
    except Exception as e:
        print(f"Error calculating directory size: {str(e)}")
    return total_size

@files_bp.route('/download-model', methods=['POST'])
def download_model():
    """Download a model from Hugging Face"""
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        
        if not model_name:
            return jsonify({'error': 'Model name is required'}), 400
        
        # Start download in background thread
        download_thread = threading.Thread(
            target=download_model_background,
            args=(model_name,)
        )
        download_thread.daemon = True
        download_thread.start()
        
        return jsonify({
            'message': f'Download started for model: {model_name}',
            'model_name': model_name,
            'status': 'downloading'
        })
        
    except Exception as e:
        print(f"Error starting model download: {str(e)}")
        return jsonify({'error': str(e)}), 500

def download_model_background(model_name):
    """Background function to download model"""
    try:
        from huggingface_hub import snapshot_download
        
        print(f"Starting download of model: {model_name}")
        
        # Download the model
        snapshot_download(
            repo_id=model_name,
            resume_download=True,
            local_files_only=False
        )
        
        print(f"Successfully downloaded model: {model_name}")
        
    except Exception as e:
        print(f"Error downloading model {model_name}: {str(e)}")
