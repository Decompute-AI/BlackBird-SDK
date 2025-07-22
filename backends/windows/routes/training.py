"""
Training routes module for Decompute Windows backend
Handles model fine-tuning, training jobs, and agent initialization
"""

from flask import Blueprint, request, jsonify, Response
import os
import json
import threading
import time
import tempfile
import shutil
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from pathlib import Path

import uuid
# Import from core modules
from core.config import (
    BASE_UPLOAD_FOLDER, ALLOWED_EXTENSIONS, UPLOAD_FOLDER,
    training_history_file, MIN_TRAINING_INTERVAL_HOURS,
    IDLE_THRESHOLD_SECONDS, TRAINING_CHECK_INTERVAL, DOCUMENTS_DIR, MODEL_WEIGHTS_DIR, MODEL_MEMORY_DIR, global_state
)

# Import RAG functionality
try:
    from rag_chat import RAGChat, create_training_files_with_feedback, run_training_with_derived_paths
except ImportError:
    RAGChat = None
    create_training_files_with_feedback = None
    run_training_with_derived_paths = None

# Import fine-tuning modules
try:
    from lora2 import load_model_and_tokenizer, apply_lora, load_datasets, load_adapter, train_model, save_adapter, generate, loss
    import torch.optim as optim
except ImportError:
    load_model_and_tokenizer = None
    apply_lora = None
    load_datasets = None
    load_adapter = None
    train_model = None
    save_adapter = None
    generate = None
    loss = None
    optim = None

try:
    from finetune_schedule import AgentFineTuner
except ImportError:
    AgentFineTuner = None

try:
    import train_agent_initial
except ImportError:
    train_agent_initial = None

# Create blueprint
training_bp = Blueprint('training', __name__)

# Global variables for training
active_jobs = {}
active_processes = {}
training_lock = threading.Lock()
is_training = False

def get_agent_folder(agent_id, storage_type):
    """Get the appropriate folder path for an agent and storage type"""
    return os.path.join(BASE_UPLOAD_FOLDER, agent_id, storage_type)

def get_unique_directory(base_path, proposed_name, force_unique=True):
    """
    Return a directory path under `base_path` based on `proposed_name`.
    If force_unique is False, returns the exact path without checking for uniqueness.
    """
    candidate = os.path.join(base_path, proposed_name)
    if not force_unique:
        return candidate
    
    if not os.path.exists(candidate):
        return candidate
    
    idx = 1
    while True:
        new_candidate = os.path.join(base_path, f"{proposed_name}_{idx}")
        if not os.path.exists(new_candidate):
            return new_candidate
        idx += 1

def process_file_path(file_path, destination_folder):
    """Helper function to process a single file path"""
    if os.path.exists(file_path):
        original_filename = os.path.basename(file_path)
        name_part, ext_part = os.path.splitext(original_filename)
        if ext_part.lower() in ALLOWED_EXTENSIONS:
            clean_name = clean_filename(name_part) + ext_part
            destination_path = os.path.join(destination_folder, clean_name)
            
            # Handle duplicate filenames
            counter = 1
            while os.path.exists(destination_path):
                destination_path = os.path.join(
                    destination_folder,
                    f"{clean_filename(name_part)}_{counter}{ext_part}"
                )
                counter += 1
            
            shutil.copy2(file_path, destination_path)
            return os.path.basename(destination_path)
    return None

def clean_filename(filename):
    """Create a clean, readable version of the filename without special characters."""
    name = os.path.splitext(filename)[0]
    clean_name = ''.join(c if c.isalnum() else '_' for c in name)
    return clean_name

def get_agent_directories(agent_type):
    """Create and return agent-specific directories"""
    agent_docs_dir = os.path.join(DOCUMENTS_DIR, agent_type)
    agent_weights_dir = os.path.join(MODEL_WEIGHTS_DIR, agent_type)
    agent_memory_dir = os.path.join(MODEL_MEMORY_DIR, agent_type)
    
    for directory in [agent_docs_dir, agent_weights_dir, agent_memory_dir]:
        os.makedirs(directory, exist_ok=True)
        
    return agent_docs_dir, agent_weights_dir, agent_memory_dir

def run_finetuning(args):
    """Run fine-tuning with the specified arguments"""
    model_name = args.get('model')
    iters = args.get('iters', 1)
    pdf_filepath = args.get('pdf_filepath', '')
    learning_rate = args.get('learning_rate', 1e-5)
    
    adapter_file = os.path.join(
        pdf_filepath,
        "adapters_newestest.npz"
    ) if pdf_filepath else "adapters_newestest.npz"
    args['adapter_file'] = adapter_file
    tokenizer_config = {"add_eos_token": False}
    args['resume_adapter_file'] = adapter_file
    
    try:
        # Validate the data directory exists
        data_dir = args.get('data', '')
        if not data_dir or not os.path.exists(data_dir):
            yield f"data: {json.dumps({'message': f'Data directory not found: {data_dir}', 'status': 'error'})}\n\n"
            return
            
        # Check if train.jsonl and valid.jsonl exist
        train_file = os.path.join(data_dir, 'train.jsonl')
        valid_file = os.path.join(data_dir, 'valid.jsonl')
        
        if not os.path.exists(train_file):
            yield f"data: {json.dumps({'message': f'Training file not found: {train_file}', 'status': 'error'})}\n\n"
            return
            
        if not os.path.exists(valid_file):
            yield f"data: {json.dumps({'message': f'Validation file not found: {valid_file}', 'status': 'error'})}\n\n"
            return
        
        yield f"data: {json.dumps({'message': 'Loading Model'})}\n\n"
        model, tokenizer = load_model_and_tokenizer(model_name, tokenizer_config)
        # LoRA is already applied by Unsloth's get_peft_model
        
        # Load datasets with error handling
        try:
            yield f"data: {json.dumps({'message': 'Loading Datasets'})}\n\n"
            train_set, valid_set = load_datasets(args)
            
            # Check if datasets are valid for training
            if len(train_set) == 0 or len(valid_set) == 0:
                yield f"data: {json.dumps({'message': 'Insufficient data for fine-tuning. Check that your training files exist and contain valid data.', 'status': 'error'})}\n\n"
                return
                
            # Training setup
            yield f"data: {json.dumps({'message': 'Setting up training'})}\n\n"
            opt = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Run training with yield forwarding
            for update in train_model(model, train_set, valid_set, opt, loss, tokenizer, args):
                yield f"data: {json.dumps(update)}\n\n"
                
                # If training was aborted due to dataset issues, stop here
                if update.get('status') == 'error' and 'dataset' in update.get('message', '').lower():
                    return
            
            # Save adapter
            # save_adapter(model, adapter_file)
            yield f"data: {json.dumps({'message': 'Fine-tuning completed successfully'})}\n\n"
            
        except FileNotFoundError as e:
            print(f"Dataset file not found: {str(e)}")
            yield f"data: {json.dumps({'message': f'Dataset files not found: {str(e)}. Please check that your training files exist.', 'status': 'error'})}\n\n"
            return
            
        except Exception as e:
            print(f"Error during dataset loading or training: {str(e)}")
            yield f"data: {json.dumps({'message': f'Error during dataset loading or training: {str(e)}', 'status': 'error'})}\n\n"
            return
            
    except Exception as e:
        print(f"Error in finetuning process: {str(e)}")
        yield f"data: {json.dumps({'message': f'Finetuning process failed: {str(e)}', 'status': 'error'})}\n\n"
        return

def run_finetune_job(job_id, file_path, model_name, epochs, learning_rate):
    """Background function to run the fine-tuning process"""
    try:
        job = active_jobs[job_id]
        job['status'] = 'processing'
        
        # Step 1: Determine output directory for training files
        model_folder = file_path  # Might be file or folder
        
        # Step 2: Create training files with feedback
        job['progress'].append({
            'time': time.time(),
            'message': 'Creating training files with feedback data'
        })
        
        output_dir = run_training_with_derived_paths(model_folder=model_folder)
        
        job['progress'].append({
            'time': time.time(),
            'message': 'Training files created successfully'
        })
        
        # Step 3: Configure fine-tuning arguments
        args = {
            "model": model_name,
            "data": output_dir,
            "iters": epochs,
            "learning_rate": learning_rate,
            "batch_size": 4,
            "steps_per_eval": 2,
            "steps_per_report": 1,
            "save_every": epochs,
            "pdf_filepath": model_folder
        }
        
        # Step 4: Run fine-tuning
        job['progress'].append({
            'time': time.time(),
            'message': 'Starting fine-tuning process'
        })
        
        for output in run_finetuning(args):
            # Expected output format: "data: {...}\n\n"
            if output.startswith('data: '):
                try:
                    data = json.loads(output[6:].strip())
                    job['progress'].append({
                        'time': time.time(),
                        **data
                    })
                except json.JSONDecodeError:
                    job['progress'].append({
                        'time': time.time(),
                        'message': output[6:].strip()
                    })
        
        job['status'] = 'completed'
        job['progress'].append({
            'time': time.time(),
            'message': 'Fine-tuning completed successfully'
        })
        
    except Exception as e:
        job = active_jobs[job_id]
        job['status'] = 'failed'
        job['error'] = str(e)
        job['progress'].append({
            'time': time.time(),
            'error': str(e)
        })
        print(f"Error in fine-tuning job {job_id}: {str(e)}")

def background_finetune(pdf_filepath, model_name, epochs, learning_rate, agent_type, storage_type, finance_toggle=False):
    """Optimized background fine-tuning process with improved error handling"""
    
    def progress_generator():
        try:
            global is_training
            global rag_chat
            
            # Validate inputs
            if not model_name:
                raise ValueError("Model name is required")
            if not pdf_filepath:
                raise ValueError("PDF filepath is required")
                
            # Log parameters for debugging
            print(f"Starting fine-tuning with parameters:")
            print(f"Model: {model_name}")
            print(f"PDF path: {pdf_filepath}")
            print(f"Epochs: {epochs}")
            print(f"Learning rate: {learning_rate}")
            
            yield f"data: {json.dumps({'message': f'Using selected model for further processing'})}\n\n"
            yield f"data: {json.dumps({'message': f'Starting processing of file'})}\n\n"
            
            # Initialize RAG chat
            try:
                is_training = False
                rag_chat = initialize_rag_chat(
                    model_name=model_name,
                    pdf_path=pdf_filepath,
                    use_finetuning=is_training,
                    agent=agent_type,
                    finance_toggle=finance_toggle
                )
                
                if not rag_chat:
                    raise ValueError("Failed to initialize RAG chat instance")
                    
            except Exception as e:
                yield f"data: {json.dumps({'error': f'RAG initialization failed: {str(e)}'})}\n\n"
                raise
            
            # This is just an example of how you might store processed JSON
            json_output_dir = os.path.join(
                UPLOAD_FOLDER,
                os.path.basename(pdf_filepath).rsplit('.', 1)[0]
            )
            # Process PDF
            try:
                if '.wav' in pdf_filepath:
                    yield f"data: {json.dumps({'message': 'Voice is getting processed'})}\n\n"

                rag_chat.process_input(pdf_filepath)
                yield f"data: {json.dumps({'message': 'Processing of file complete'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': f'Processing of file failed: {str(e)}'})}\n\n"
                raise
            
            # Configure fine-tuning arguments
            args = {
                "model": model_name,
                "data": json_output_dir,
                "iters": epochs,
                "learning_rate": learning_rate,
                "batch_size": 4,
                "steps_per_eval": 2,
                "steps_per_report": 1,
                "save_every": epochs,
                "pdf_filepath": pdf_filepath
            }
            
            # Run fine-tuning
            
            try:
                training_skipped = False  # Track if training was skipped
                for output in run_finetuning(args):
                    yield output
                    
                    # NEW – Check if training was skipped due to OOM
                    if output.startswith('data: '):
                        try:
                            data = json.loads(output[6:].strip())
                            if data.get('status') == 'warning':
                                print("Fine-tuning was skipped; continuing with RAG-only mode")
                                training_skipped = True
                                break
                        except json.JSONDecodeError:
                            pass  # Continue normally if JSON parsing fails
                
                # Only proceed with success messages if training wasn't skipped
                if not training_skipped:
                    yield f"data: {json.dumps({'message': 'Fine-tuned model loaded successfully'})}\n\n"
                else:
                    yield f"data: {json.dumps({'message': 'Continuing with RAG-only mode (training skipped)'})}\n\n"
                    
            except Exception as e:
                yield f"data: {json.dumps({'error': f'Fine-tuning error: {str(e)}'})}\n\n"
                training_skipped = True  # Mark training as skipped if it failed
                raise
            
            # Determine whether to use fine-tuning based on training success
            use_finetuning_for_model = not training_skipped
            
            if use_finetuning_for_model:
                yield f"data: {json.dumps({'message': 'Fine-tuned model loaded successfully'})}\n\n"
            
            yield f"data: {json.dumps({'message': 'Begin asking your question'})}\n\n"
            
            # ──────────────────────────────────────────────────────────────────────────────────
            # CRITICAL: Clear all GPU memory before loading model for inference
            # ──────────────────────────────────────────────────────────────────────────────────
            print("Clearing GPU memory before model loading...")
            import torch
            import gc
            
            # Force garbage collection
            gc.collect()
            
            # Clear PyTorch CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all operations to complete
                print(f"GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()} bytes")
            
            yield f"data: {json.dumps({'message': 'Memory cleared, initializing model...'})}\n\n"
            
            # Initialize the model with appropriate finetuning flag
            is_training = use_finetuning_for_model
            rag_chat = initialize_rag_chat(
                model_name=model_name,
                pdf_path=pdf_filepath,
                use_finetuning=is_training,
                agent=agent_type,
                finance_toggle=finance_toggle
            )
            
            # Signal completion
            yield f"data: {json.dumps({'status': 'complete'})}\n\n"
            
            # --------------------------------------------------------
            # DELETE ALLOWED FILES FROM THE PDF's DIRECTORY (Cleanup)
            # --------------------------------------------------------
            dir_to_clean = os.path.dirname(pdf_filepath)
            try:
                for root, dirs, files in os.walk(dir_to_clean):
                    for f in files:
                        ext = os.path.splitext(f)[1].lower()
                        if ext in ALLOWED_EXTENSIONS:
                            if (f == "chunk_mapping.json" or f== "conversation.json" or f=="extracted_ngrams.json" or f=="config.json" or f=="feedback.json"):
                                continue

                            file_path = os.path.join(root, f)
                            os.remove(file_path)
                            print(f"Deleted file: {file_path}")
            except Exception as cleanup_error:
                # Not critical to block everything if cleanup fails,
                # but you can yield or log an error here if desired.
                print(f"Error while cleaning up files: {cleanup_error}")
                yield f"data: {json.dumps({'message': f'Cleanup failed: {str(cleanup_error)}'})}\n\n"
            
        except Exception as e:
            error_message = f"Error during fine-tuning process: {str(e)}"
            yield f"data: {json.dumps({'error': error_message})}\n\n"
            is_training = False
            if 'rag_chat' in globals():
                rag_chat = None
            raise

    return progress_generator

def get_idle_time():
    """Get system idle time in seconds"""
    try:
        # This is a Windows-specific implementation
        import ctypes
        
        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [
                ('cbSize', ctypes.c_uint),
                ('dwTime', ctypes.c_uint),
            ]
        
        lastInputInfo = LASTINPUTINFO()
        lastInputInfo.cbSize = ctypes.sizeof(lastInputInfo)
        ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lastInputInfo))
        millis = ctypes.windll.kernel32.GetTickCount() - lastInputInfo.dwTime
        return millis / 1000.0  # Convert to seconds
    except Exception as e:
        print(f"Error getting idle time: {e}")
        return 0

def get_training_history():
    """Load the training history from file"""
    try:
        os.makedirs(os.path.dirname(training_history_file), exist_ok=True)
        
        if os.path.exists(training_history_file):
            with open(training_history_file, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading training history: {e}")
        return {}

def update_training_history(agent_name):
    """Update the training history with the current time"""
    try:
        history = get_training_history()
        history[agent_name] = time.time()
        
        with open(training_history_file, 'w') as f:
            json.dump(history, f)
            
        print(f"Updated training history for {agent_name}")
    except Exception as e:
        print(f"Error updating training history: {e}")

def should_train_agent(agent_name):
    """Check if an agent should be trained based on the last training time"""
    history = get_training_history()
    last_time = history.get(agent_name, 0)
    current_time = time.time()
    
    # Check if enough time has passed
    hours_since_last = (current_time - last_time) / 3600
    return hours_since_last >= MIN_TRAINING_INTERVAL_HOURS

def get_agents_for_training():
    """Get a list of agents that need training, sorted by priority"""
    try:
        # This assumes you have an internal function to get all agents
        # Replace with your actual method of getting agent names        
        # Alternative: hardcoded list if you don't have a function
        agents = ['general', 'research', 'legal']
        
        # Filter and sort by last training time
        history = get_training_history()
        
        trainable_agents = []
        for agent in agents:
            if should_train_agent(agent):
                last_time = history.get(agent, 0)
                trainable_agents.append((agent, last_time))
        
        # Sort by last training time, oldest first
        trainable_agents.sort(key=lambda x: x[1])
        
        # Return just the agent names
        return [agent for agent, _ in trainable_agents]
    except Exception as e:
        print(f"Error getting agents for training: {e}")
        return ['general']  # Default to 'general' on error

def run_finetuning_background(agent_name):
    """Run the finetuning process for an agent"""
    global is_training
    
    # Create date string for 2 days ago
    two_days_ago = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    
    try:
        # Get your AgentFineTuner class instance
        # This assumes your AgentFineTuner class is imported and available
        
        tuner = AgentFineTuner()
        print(f"Starting finetuning for agent: {agent_name}")
        
        results = tuner.finetune_agent(
            agent_name,
            epochs_increment=25,
            learning_rate=1e-5,
            after_date=two_days_ago
        )
        
        print("\nFine-tuning results:")
        for result in results:
            status = "SUCCESS" if result.get("success") else "FAILED"
            print(f"{status}: {result.get('location')}")
        
        # Check if any training was successful
        if any(result.get("success", False) for result in results):
            update_training_history(agent_name)
            return True
        return False
    except Exception as e:
        print(f"Error running finetuning: {e}")
        return False
    finally:
        # Always make sure to reset the training flag
        with training_lock:
            is_training = False
        
        # Run garbage collection to free memory
        import gc
        gc.collect()

def background_training_monitor():
    """Background thread to monitor idle time and initiate training when appropriate"""
    print("Starting background training monitor")
    
    while True:
        try:
            # Sleep at the beginning to prevent immediate training at startup
            time.sleep(TRAINING_CHECK_INTERVAL)
            
            # Skip if already training
            with training_lock:
                if is_training:
                    continue
            
            # Get system idle time
            idle_time = get_idle_time()
            # print(f"Current idle time: {idle_time:.2f} seconds")
            
            # Check if system is idle enough
            if idle_time > IDLE_THRESHOLD_SECONDS:
                print("System is idle, checking for agents to train")
                
                # Get agents that need training
                agents = get_agents_for_training()
                
                if agents:
                    agent_to_train = agents[0]  # Train the highest priority agent
                    print(f"Selected agent for training: {agent_to_train}")
                    
                    # Set the training flag
                    with training_lock:
                        is_training = True
                    
                    # Start training in a separate thread to avoid blocking
                    training_thread = threading.Thread(
                        target=run_finetuning_background,
                        args=(agent_to_train,)
                    )
                    training_thread.daemon = True
                    training_thread.start()
                else:
                    print("No agents need training at this time")
        except Exception as e:
            print(f"Error in background training monitor: {e}")

def initialize_rag_chat(model_name, pdf_path, use_finetuning, agent, finance_toggle=False):
    """Initialize RAG chat with the specified parameters"""
    try:
        import torch
        
        # Handle SDK initialization with temporary files
        if pdf_path and os.path.exists(pdf_path):
            if "sdk_temp_" in pdf_path:
                print(f"[SDK] Initializing RAG chat for SDK with temp directory: {pdf_path}")
                
                temp_files = [f for f in os.listdir(pdf_path) if f.endswith('.txt')]
                if not temp_files:
                    temp_init_file = os.path.join(pdf_path, "default_init.txt")
                    with open(temp_init_file, 'w', encoding='utf-8') as f:
                        f.write(f"Default initialization file for {agent} agent using SDK.")
                    print(f"[SDK] Created default initialization file: {temp_init_file}")
        
        return RAGChat(
            model_name=model_name,
            pdf_path=pdf_path,
            chunk_size=1000,
            chunk_overlap=200,
            max_chunks=3,
            use_finetuning=use_finetuning,
            agent=agent,
            finance_toggle=finance_toggle
        )
    except Exception as e:
        print(f"Error in initialize_rag_chat: {str(e)}")
        
        if "out of memory" in str(e).lower() or "memory" in str(e).lower():
            print("[WARN] RAG init failed due to OOM – retrying with no fine-tune")
            try:
                return RAGChat(
                    model_name=model_name,
                    pdf_path=pdf_path,
                    chunk_size=1000,
                    chunk_overlap=200,
                    max_chunks=3,
                    use_finetuning=False,
                    agent=agent,
                    finance_toggle=finance_toggle
                )
            except Exception as fallback_e:
                print(f"[ERROR] Even RAG-only mode failed: {fallback_e}")
                raise RuntimeError(f"Failed to initialize RAG chat even in fallback mode: {str(fallback_e)}")
        
        raise RuntimeError(f"Failed to initialize RAG chat: {str(e)}")

@training_bp.route('/api/finetune', methods=['POST'])
def start_finetuning():
    """
    API endpoint to start fine-tuning with feedback incorporated
    
    Expected JSON payload:
    {
        "file_path": "/path/to/file/or/model/folder",
        "model_name": "name-of-model", 
        "epochs": 3,
        "learning_rate": 1e-5
    }
    
    Returns:
        - 202 Accepted with job_id for tracking
        - 400 Bad Request if parameters are missing
        - 500 Internal Server Error if another error occurs
    """
    try:
        data = request.json
        
        # Validate required parameters
        if not data or 'file_path' not in data or 'model_name' not in data:
            return jsonify({
                'error': 'Missing required parameters. Need file_path and model_name.'
            }), 400
        
        # Extract parameters
        file_path = data['file_path']
        model_name = data['model_name']
        epochs = data.get('epochs', 3)  # Default to 3 epochs
        learning_rate = data.get('learning_rate', 1e-5)  # Default learning rate
        
        # Validate file_path exists
        if not os.path.exists(file_path):
            return jsonify({
                'error': f'Path not found: {file_path}'
            }), 400
            
        # Generate a unique job ID
        job_id = f"finetune_{int(time.time())}"
        
        # Create a background thread for the fine-tuning process
        thread = threading.Thread(
            target=run_finetune_job,
            args=(job_id, file_path, model_name, epochs, learning_rate)
        )
        thread.daemon = True
        
        # Track the job
        active_jobs[job_id] = {
            'status': 'starting',
            'file_path': file_path,
            'model_name': model_name,
            'start_time': time.time(),
            'progress': [],
            'error': None
        }
        
        # Start the thread
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'accepted',
            'message': 'Fine-tuning job started in the background'
        }), 202
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to start fine-tuning job: {str(e)}'
        }), 500

@training_bp.route('/api/finetune/status/<job_id>', methods=['GET'])
def get_finetune_status(job_id):
    """Get the status of a fine-tuning job"""
    if job_id not in active_jobs:
        return jsonify({
            'error': 'Job not found'
        }), 404
    
    job = active_jobs[job_id]
    
    return jsonify({
        'job_id': job_id,
        'status': job['status'],
        'file_path': job['file_path'],
        'model_name': job['model_name'],
        'start_time': job['start_time'],
        'elapsed': time.time() - job['start_time'],
        'progress': job['progress'][-10:] if job['progress'] else [],  # Return last 10 progress updates
        'error': job['error']
    })

@training_bp.route('/initialize-rag', methods=['POST'])
def initialize_rag():
    """Initialize RAG with uploaded files and configuration"""
    import tempfile
    try:
        # Extract and validate required parameters
        model_name = request.form.get('model')
        agent_type = request.form.get('agent_type')
        storage_type = request.form.get('storage_type')

        finance_toggle = request.form.get('finance_toggle', False)
        custom_filename = request.form.get('custom_filename')

        if not all([model_name, agent_type, storage_type]):
            return jsonify({
                "error": "Missing required parameters. Please provide model, agent_type, and storage_type"
            }), 400

        # Validate agent type and storage type
        valid_agents = {'tech', 'legal', 'finance', 'meetings', 'general', 'research'}
        valid_storage = {'model_memory', 'saved_files'}

        if agent_type not in valid_agents:
            return jsonify({"error": f"Invalid agent type. Must be one of: {', '.join(valid_agents)}"}), 400
        if storage_type not in valid_storage:
            return jsonify({"error": f"Invalid storage type. Must be one of: {', '.join(valid_storage)}"}), 400

        epochs = int(request.form.get('epochs', 10))
        learning_rate = float(request.form.get('learning_rate', 5e-5))

        # Create a unique process ID
        process_id = str(uuid.uuid4())

        # Define base path for this agent and storage type
        base_storage_path = os.path.join(BASE_UPLOAD_FOLDER, agent_type, storage_type)

        processed_files = []
        file_sources = []

        # Get all possible file sources
        files_from_form = request.form.getlist('files[]')
        files_from_upload = request.files.getlist('files[]')
        folder_path = request.form.get('folder_path')

        print(f"Files from form: {files_from_form}")  # Debug log
        print(f"Files from upload: {files_from_upload}")  # Debug log
        print(f"Folder path: {folder_path}")  # Debug log

        # Process uploaded files
        if files_from_upload:
            for file in files_from_upload:
                if file and file.filename:
                    # Debug logging
                    print(f"Debug - Original filename: {file.filename}")
                    safe_name = secure_filename(file.filename)
                    print(f"Debug - Safe filename: {safe_name}")
                    
                    # Create temporary directory and save file
                    temp_dir = tempfile.mkdtemp()
                    temp_path = os.path.join(temp_dir, safe_name)
                    print(f"Debug - Temp path: {temp_path}")
                    
                    file.save(temp_path)
                    file_sources.append({
                        'path': temp_path,
                        'original_name': file.filename,
                        'safe_name': safe_name
                    })

        # Process file paths from form
        if files_from_form:
            for filepath in files_from_form:
                if filepath and os.path.exists(filepath):
                    _, ext = os.path.splitext(filepath)
                    if ext.lower() in ALLOWED_EXTENSIONS:
                        file_sources.append(filepath)
                        print(f"Added file source: {filepath}")  # Debug log

        print(f"Collected file sources: {file_sources}")  # Debug log

        # Set up destination folder
        if storage_type == 'model_memory':
            folder_name = f"{agent_type}_combined_weights"
            destination_folder = get_unique_directory(base_storage_path, folder_name, force_unique=False)
        else:
            if file_sources:
                if len(file_sources) == 1:
                    file_info = file_sources[0]
                    if isinstance(file_info, dict):
                        name_part = os.path.splitext(file_info['original_name'])[0]
                    else:
                        name_part = os.path.splitext(os.path.basename(file_info))[0]
                    folder_name = clean_filename(name_part)
                else:
                    first_file = file_sources[0]
                    if isinstance(first_file, dict):
                        name_part = os.path.splitext(first_file['original_name'])[0]
                    else:
                        name_part = os.path.splitext(os.path.basename(first_file))[0]
                    others_count = len(file_sources) - 1
                    folder_name = f"{clean_filename(name_part)}_{others_count}_others"
            else:
                folder_name = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            destination_folder = get_unique_directory(base_storage_path, folder_name, force_unique=True)

        # Create the destination folder
        os.makedirs(destination_folder, exist_ok=True)
        print(f"Created destination folder: {destination_folder}")  # Debug log

        # Process all collected file sources
        for file_source in file_sources:
            try:
                if isinstance(file_source, dict):
                    source_path = file_source['path']
                    destination_path = os.path.join(destination_folder, file_source['safe_name'])
                else:
                    source_path = file_source
                    destination_path = os.path.join(destination_folder, os.path.basename(source_path))

                if os.path.exists(source_path):
                    shutil.copy2(source_path, destination_path)
                    processed_files.append(os.path.basename(destination_path))
                    print(f"Successfully copied {source_path} to {destination_path}")  # Debug log
            except Exception as e:
                print(f"Error processing file {source_path}: {str(e)}")  # Debug log
                continue

        # Clean up temporary files
        for file_source in file_sources:
            if isinstance(file_source, dict):
                temp_dir = os.path.dirname(file_source['path'])
                if tempfile.gettempdir() in temp_dir:
                    try:
                        os.remove(file_source['path'])
                        os.rmdir(temp_dir)
                    except:
                        pass

        # Store process parameters
        active_processes[process_id] = {
            'model_name': model_name,
            'agent_type': agent_type,
            'storage_type': storage_type,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'input_path': destination_folder,
            'finance_toggle': finance_toggle,
            'status': 'initialized'
        }

        config = {
            "agent_type": agent_type,
            "model": model_name,
            "date": datetime.now().isoformat(),
            "display_name": custom_filename if custom_filename else "Untitled Document",
            "finance_toggle": finance_toggle if agent_type == "finance" else None
        }
        
        with open(os.path.join(destination_folder, 'config.json'), 'w') as f:
            json.dump(config, f)

        return jsonify({
            "message": "Processing successful",
            "path": destination_folder,
            "process_id": process_id,
            "agent_type": agent_type,
            "storage_type": storage_type,
            "processed_files": processed_files,
            "file_sources": file_sources  # Added for debugging
        }), 200

    except Exception as e:
        print(f"Error in initialize_rag: {str(e)}")  # Debug log
        return jsonify({"error": f"Initialization failed: {str(e)}"}), 500

@training_bp.route('/fine-tuning-progress')
def fine_tuning_progress():
    """Stream fine-tuning progress updates"""
    process_id = request.args.get('process_id')
    
    if not process_id or process_id not in active_processes:
        return jsonify({"error": "Invalid or expired process ID"}), 400
    
    process_data = active_processes[process_id]
    filepath = process_data['input_path']
    
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    
    progress_generator = background_finetune(
        pdf_filepath=filepath,
        model_name=process_data['model_name'],
        epochs=process_data['epochs'],
        learning_rate=process_data['learning_rate'],
        agent_type=process_data['agent_type'],
        storage_type=process_data['storage_type'],
        finance_toggle=process_data['finance_toggle']
    )

    if process_id in active_processes:
        del active_processes[process_id]
    
    return Response(
        progress_generator(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        }
    )

@training_bp.route('/load-existing-model', methods=['POST'])
def load_existing_model():
    """Load an existing fine-tuned model"""
    try:
        import torch  # Import torch directly in the function
        
        data = request.get_json()
        
        filename = data.get('filename')
        model_name = data.get('modelname')  # <-- retrieve the model name
        agent = data.get('agent')
        finance_toggle = data.get('finance_toggle', "false")  # Get finance toggle state with default
        response_data = {
            'success': True,
            'message': 'Model loaded successfully'
        }

        if not filename:
            return jsonify({'error': 'Filename is required'}), 400

        if not model_name:
            return jsonify({'error': 'Model name is required'}), 400
        
        folder_path = os.path.join(filename)

        if not os.path.exists(folder_path):
            return jsonify({'error': 'Model not found'}), 404

        # Only handle finance toggle for finance agent
        if agent == 'finance':
            print(f"Loading finance agent with initial toggle state: {finance_toggle}")
            # Load config to get stored finance toggle state
            config_path = os.path.join(folder_path, 'config.json')
            try:
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        # Use stored finance toggle state if it exists
                        stored_toggle = config.get('finance_toggle')
                        if stored_toggle is not None:
                            finance_toggle = stored_toggle
                            print(f"Found stored finance toggle state: {finance_toggle}")
                            response_data['finance_toggle'] = finance_toggle
                            response_data['message'] = f'Model loaded successfully with finance toggle: {finance_toggle}'
                        else:
                            print("No stored finance toggle state found, using default")
                            response_data['finance_toggle'] = finance_toggle
                            response_data['message'] = f'Model loaded with default finance toggle: {finance_toggle}'
            except Exception as e:
                print(f"Error reading config file: {str(e)}, using default finance toggle state")
                response_data['finance_toggle'] = finance_toggle
                response_data['message'] = f'Model loaded with default finance toggle (error reading config): {finance_toggle}'
        else:
            # For non-finance agents, don't pass the finance_toggle parameter
            rag_chat = initialize_rag_chat(
                model_name=model_name,
                pdf_path=folder_path,
                use_finetuning=True,
                load_existing=True,
                agent=agent
            )
            return jsonify(response_data)

        # For finance agent, include the finance_toggle parameter
        rag_chat = initialize_rag_chat(
            model_name=model_name,
            pdf_path=folder_path,
            use_finetuning=True,
            load_existing=True,
            agent=agent,
            finance_toggle=finance_toggle
        )
        print(f"Finance agent initialized with toggle state: {finance_toggle}")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in load_existing_model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@training_bp.route('/model-details', methods=['GET'])
def get_model_details():
    """Get details about a specific model"""
    try:
        model_name = request.args.get('model')
        if not model_name:
            return jsonify({'error': 'Model name is required'}), 400
        
        # Get cache info
        from huggingface_hub import scan_cache_dir
        hf_cache_info = scan_cache_dir()
        
        # Find the requested model in the cache
        model_info = None
        for repo in hf_cache_info.repos:
            if repo.repo_id == model_name and repo.repo_type == 'model':
                model_info = repo
                break
        
        if not model_info:
            return jsonify({'error': 'Model not found in cache'}), 404
        
        # Read model config
        config_path = os.path.join(model_info.repo_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Convert last_modified to string if it exists
        last_modified = None
        if model_info.last_modified is not None:
            if isinstance(model_info.last_modified, (int, float)):
                last_modified = datetime.fromtimestamp(model_info.last_modified).isoformat()
            else:
                last_modified = model_info.last_modified.isoformat()
        
        return jsonify({
            'name': model_info.repo_id,
            'path': str(model_info.repo_path),
            'size': model_info.size_on_disk,
            'last_modified': last_modified,
            'config': config
        })
        
    except Exception as e:
        print(f"Error in get_model_details: {str(e)}")  # Add debugging
        return jsonify({'error': str(e)}), 500

@training_bp.route('/api/cleanup', methods=['POST'])
def cleanup():
    """
    Endpoint to release memory and unload models.
    This is called when the Electron app is closing.
    """
    try:
        # Store references to your models
        global rag_chat, flux_pipeline
        rag_chat = None
        rag = RAGChat
        rag.reset_instance()
        del rag
        del rag_chat
        print("cleared all memory")
        
        # Clear image generation model if loaded
        if 'flux_pipeline' in globals() and flux_pipeline is not None:
            del flux_pipeline
            flux_pipeline = None
            print("cleared image generation model")
        
        # Clear PyTorch CUDA memory if available
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("PyTorch CUDA cache cleared")
        
        # Force garbage collection
        import gc
        gc.collect()

        return jsonify({
            'success': True, 
            'message': 'Successfully unloaded models and cleared caches',
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error during cleanup: {str(e)}'
        }), 500

@training_bp.route('/api/save-feedback', methods=['POST'])
def save_feedback():
    """Save user feedback for model improvement"""
    try:
        data = request.json        
        # Validate required fields
        required_fields = ['message_id', 'assistant_message', 'is_positive', 'agent_type']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        agent_type = data['agent_type']
        file_path = data.get('file_path', '')
        
        if not file_path:
            return jsonify({'error': 'file_path is required'}), 400
            
        # Create a feedback data structure
        feedback_data = {
            'assistant_message': data['assistant_message'],
            'is_positive': data['is_positive'],
            'agent_type': agent_type,
            'model': data.get('model', 'unknown'),
            'timestamp': data.get('timestamp', datetime.now().isoformat())
        }
        
        # Create the feedback file path
        feedback_file = os.path.join(file_path, 'feedback.json')
        
        try:
            # Check if the feedback file already exists
            if os.path.isfile(feedback_file):
                # Read existing feedback data
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    try:
                        existing_feedback = json.load(f)
                        print(f"Successfully loaded existing feedback from {feedback_file}")
                        
                        # If the structure is just an array
                        if isinstance(existing_feedback, list):
                            existing_feedback.append(feedback_data)
                        else:
                            # If the structure has a 'feedback' key
                            if 'feedback' not in existing_feedback:
                                existing_feedback['feedback'] = []
                            existing_feedback['feedback'].append(feedback_data)
                            
                    except json.JSONDecodeError:
                        # If the file isn't valid JSON, start with a new array
                        print(f"File at {feedback_file} is not valid JSON. Starting with new array.")
                        existing_feedback = [feedback_data]
                
                # Write back the updated data
                with open(feedback_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_feedback, f, ensure_ascii=False, indent=2)
                print(f"Successfully appended feedback to existing file: {feedback_file}")
            else:
                # Create a new feedback file
                with open(feedback_file, 'w', encoding='utf-8') as f:
                    # Use a simple array structure to match conversation.json pattern
                    json.dump([feedback_data], f, ensure_ascii=False, indent=2)
                print(f"Created new feedback file at: {feedback_file}")
        except Exception as e:
            print(f"Error processing feedback file {feedback_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f"Error processing feedback file: {str(e)}"}), 500
        
        # Also save to a central feedback log for easier analysis
        try:
            # Create base feedback directory if it doesn't exist
            base_feedback_dir = os.path.join(UPLOAD_FOLDER, 'feedback_data')
            os.makedirs(base_feedback_dir, exist_ok=True)
            
            # Create agent-specific directory
            agent_dir = os.path.join(base_feedback_dir, agent_type)
            os.makedirs(agent_dir, exist_ok=True)
            
            # Append to feedback log file
            feedback_log_path = os.path.join(base_feedback_dir, f'{agent_type}_feedback_log.jsonl')
            with open(feedback_log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(json.dumps(feedback_data) + '\n')
            print(f"Also logged feedback to central log: {feedback_log_path}")
        except Exception as e:
            print(f"Warning: Could not log to central feedback log: {str(e)}")
            # Continue despite this error - we already saved to the main file
        
        return jsonify({
            'success': True, 
            'message': 'Feedback saved successfully',
            'file_path': feedback_file
        }), 200
        
    except Exception as e:
        print(f"Error saving feedback: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@training_bp.route('/api/training/trigger', methods=['POST'])
def trigger_training():
    """Endpoint to manually trigger training for an agent"""
    global is_training
    
    try:
        request_data = request.get_json()
        agent_name = request_data.get('agent_name', 'general')
        force = request_data.get('force', False)
        
        # Check if training is already in progress
        with training_lock:
            if is_training:
                return jsonify({
                    'success': False,
                    'message': 'Training already in progress'
                }), 409
        
        # Check if agent should be trained (unless forced)
        if not force and not should_train_agent(agent_name):
            last_trained = get_training_history().get(agent_name, 0)
            hours_ago = (time.time() - last_trained) / 3600
            
            return jsonify({
                'success': False,
                'message': f'Agent was trained {hours_ago:.1f} hours ago. Set force=true to override.',
                'last_trained': last_trained
            }), 400
        
        # Set the training flag
        with training_lock:
            is_training = True
        
        # Start training in a new thread
        training_thread = threading.Thread(
            target=run_finetuning_background,
            args=(agent_name,)
        )
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Training started for agent: {agent_name}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error triggering training: {str(e)}'
        }), 500

@training_bp.route('/api/training/status', methods=['GET'])
def training_status():
    """Endpoint to check if training is in progress"""
    with training_lock:
        status = is_training
    
    history = get_training_history()
    
    return jsonify({
        'training_in_progress': status,
        'last_training_times': history
    })

@training_bp.route('/api/train-initial-data', methods=['POST'])
def train_initial_data():
    """Train with initial data for agents"""
    try:
        data = request.get_json()
        training_data = data["training_data"]

        for item in training_data:
            agent_type = item["agent_type"]
            prompt = item["prompt"]
            global_state.predictor.update_user_query(agent_type, prompt)

        global_state.save_if_needed()

        return jsonify({"status": "success", "message": "Initial training data processed."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@training_bp.route('/api/initial-prompt-training', methods=['POST'])
def initial_prompt_training():
    """Load initial prompt training data"""
    try:
        train_agent_initial.train_initial()
        return jsonify({"status": "success", "message": "Loaded with initial prompt train data"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
