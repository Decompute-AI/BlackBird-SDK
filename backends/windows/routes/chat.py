"""
Chat routes module for Decompute Windows backend
Handles all chat-related functionality including conversations, history, and streaming responses
"""

from flask import Blueprint, request, jsonify, Response, stream_with_context
import os
import torch
import json
import gc
import threading
import tempfile
import shutil
from datetime import datetime
from werkzeug.utils import secure_filename

# Import from core modules
from core.config import (
    BASE_UPLOAD_FOLDER, ALLOWED_EXTENSIONS, UPLOAD_FOLDER,
    SPECIAL_CHARS
)

# Import RAG functionality
try:
    from rag_chat import RAGChat, clear_memory
except ImportError:
    RAGChat = None
    clear_memory = None

# Create blueprint
chat_bp = Blueprint('chat', __name__)

# Global variables for chat functionality
rag_chat = None
_CURRENT_MODEL = None
_CURRENT_TOKENIZER = None
_CHAT_HISTORY = []
_CURRENT_AGENT = None

def get_agent_folder(agent_id, storage_type):
    """Get the appropriate folder path for an agent and storage type"""
    return os.path.join(BASE_UPLOAD_FOLDER, agent_id, storage_type)

def get_conversation_history(dir_path):
    """Get conversation history from a directory"""
    conversation_file = os.path.join(dir_path, 'conversation.json')
    if os.path.exists(conversation_file):
        try:
            with open(conversation_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading conversation file in {dir_path}")
            return []
    return []


def clear_memory():
    """
    Frees Python references and (if supported) clears GPU cache
    to free up memory before loading a new model.
    Note: This does NOT clear conversation history.
    """
    # Initialize globals if they don't exist
    global _CURRENT_MODEL, _CURRENT_TOKENIZER, _CHAT_HISTORY, _CURRENT_AGENT
    
    if '_CURRENT_MODEL' not in globals():
        global _CURRENT_MODEL
        _CURRENT_MODEL = None
        
    if '_CURRENT_TOKENIZER' not in globals():
        global _CURRENT_TOKENIZER
        _CURRENT_TOKENIZER = None
        
    if '_CHAT_HISTORY' not in globals():
        global _CHAT_HISTORY
        _CHAT_HISTORY = []
        
    if '_CURRENT_AGENT' not in globals():
        global _CURRENT_AGENT
        _CURRENT_AGENT = None

    # Delete Python references to the old model/tokenizer if they exist
    if _CURRENT_MODEL is not None:
        try:
            del _CURRENT_MODEL
            _CURRENT_MODEL = None
        except:
            pass
            
    if _CURRENT_TOKENIZER is not None:
        try:
            del _CURRENT_TOKENIZER
            _CURRENT_TOKENIZER = None
        except:
            pass

    # Run garbage collection to free up RAM
    gc.collect()

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            print("GPU memory cleared")
        except:
            pass

def initialize_rag_chat(model_name, pdf_path, use_finetuning, agent, finance_toggle=False):
    """Initialize RAG chat with the specified parameters"""
    global rag_chat
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

def load_default_chat_model(model_name):
    """
    Clears any previously loaded model from memory, then loads and prepares 
    a new 'default chat' model using PyTorch and HuggingFace transformers.
    Returns an instance of the DefaultChatModel class.
    """
    # Initialize globals if they don't exist
    if '_CURRENT_MODEL' not in globals():
        global _CURRENT_MODEL
        _CURRENT_MODEL = None
        
    if '_CURRENT_TOKENIZER' not in globals():
        global _CURRENT_TOKENIZER
        _CURRENT_TOKENIZER = None
        
    if '_CHAT_HISTORY' not in globals():
        global _CHAT_HISTORY
        _CHAT_HISTORY = []
        
    if '_CURRENT_AGENT' not in globals():
        global _CURRENT_AGENT
        _CURRENT_AGENT = None
    
    # Offline mode, if needed, to avoid downloading
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    # Clear any previously loaded model
    clear_memory()

    try:
        # Import necessary functions
        import torch
        from unsloth import FastLanguageModel

        # Load the model and tokenizer directly using FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(model_name)
        
        # Save references for unloading next time
        _CURRENT_MODEL = model
        _CURRENT_TOKENIZER = tokenizer

        # Return the custom wrapper with existing history if available
        return DefaultChatModel(model, tokenizer)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


class DefaultChatModel:
    """
    A wrapper class that holds a reference to the model,
    provides optimized streaming generation, and tracks conversation history.
    """
    def __init__(self, model, tokenizer, max_tokens=4096, temperature=0.4):
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Try to compile model if PyTorch >= 2.1
        try:
            self.model = torch.compile(self.model)
            print("Successfully compiled model with torch.compile()")
        except Exception as e:
            print(f"Could not compile model: {e}")
        
        # Use global history and agent tracking
        global _CHAT_HISTORY, _CURRENT_AGENT
        if _CURRENT_AGENT is None:
            _CURRENT_AGENT = None
        if _CHAT_HISTORY is None:
            _CHAT_HISTORY = []

    def generate_response_stream(self, user_input, agent=None, include_history=True, system_prompt=None):
        """Streams tokens from the model's response using HF streamer and KV-cache."""
        from transformers import TextIteratorStreamer
        import threading
        import time
        
        # Add user message to history
        self.add_to_history("user", user_input)
        
        # Check if agent changed
        agent_changed = self.check_agent_change(agent)
        
        # Build messages array
        messages = self._build_messages(user_input, agent, include_history and not agent_changed)
        
        # Convert to prompt
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenize prompt (only once)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        
        # Set up streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Track generation stats
        start_time = time.time()
        total_new_tokens = 0
        
        # Start generation in a separate thread
        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            repetition_penalty=1.2,
            do_sample=True if self.temperature > 0 else False,
            use_cache=True  # Enable KV-cache
        )
        
        thread = threading.Thread(
            target=self.model.generate,
            kwargs=generation_kwargs
        )
        thread.start()
        
        # Stream the output
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            total_new_tokens += 1
            
            # Calculate tokens per second
            elapsed = time.time() - start_time
            tokens_per_second = total_new_tokens / elapsed if elapsed > 0 else 0
            
            yield new_text, tokens_per_second
        
        # Add assistant's response to history
        self.add_to_history("assistant", generated_text)
        
    def _build_messages(self, user_input, agent, include_history):
        """Build messages array for the model."""
        messages = []
        
        # Add system message based on agent
        if agent:
            messages.append({
                "role": "system",
                "content": f"""\
You are a knowledgeable and helpful {agent} AI assistant. You should:
1. Provide clear, accurate, and concise answers.
2. If you do not know the answer or cannot be certain, say so.
3. Use simple language, but include necessary details.
4. When appropriate, provide step-by-step reasoning or explanations.
5. Maintain a polite and professional tone.
6. Avoid including irrelevant or sensitive information.

Now, please answer the following user query:
"""
            })
        
        # Add conversation history if requested
        if include_history and len(_CHAT_HISTORY) > 0:
            for msg in _CHAT_HISTORY[-3:]:  # Last 3 messages
                messages.append(msg)
        
        # Add current user input
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        return messages
        
    def clear_history(self):
        """Clear the conversation history."""
        global _CHAT_HISTORY
        _CHAT_HISTORY = []

    def check_agent_change(self, agent):
        """Check if the agent has changed and clear history if needed."""
        global _CURRENT_AGENT, _CHAT_HISTORY
        
        if _CURRENT_AGENT is None:
            _CURRENT_AGENT = agent
            return False
            
        if agent != _CURRENT_AGENT:
            _CURRENT_AGENT = agent
            self.clear_history()
            return True
            
        return False
        
    def add_to_history(self, role, content):
        """Add a message to the conversation history."""
        global _CHAT_HISTORY
        _CHAT_HISTORY.append({"role": role, "content": content})


@chat_bp.route('/api/save-conversation/<agent_id>', methods=['POST'])
def save_conversation(agent_id):
    """Save conversation to agent directory"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        messages = data.get('messages', [])
        
        if not file_path:
            return jsonify({'error': 'File path is required'}), 400
            
        conversation_file = os.path.join(file_path, 'conversation.json')
        with open(conversation_file, 'w') as f:
            json.dump(messages, f)
            
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"Error saving conversation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/api/clear-conversation/<agent_id>', methods=['POST'])
def clear_conversation(agent_id):
    """Clear conversation history for an agent"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({'error': 'File path is required'}), 400
            
        conversation_file = os.path.join(file_path, 'conversation.json')
        
        if os.path.exists(conversation_file):
            os.remove(conversation_file)
            
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"Error clearing conversation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/api/chat-history/<agent_id>', methods=['GET'])
def get_agent_history(agent_id):
    """Get chat history for a specific agent"""
    try:
        history_path = get_agent_folder(agent_id, 'chat_history')
        
        if not os.path.exists(history_path):
            os.makedirs(history_path)
            return jsonify({})
            
        histories = {}
        
        for filename in os.listdir(history_path):
            if filename.endswith('.json'):
                file_path = os.path.join(history_path, filename)
                with open(file_path, 'r') as f:
                    model_path = filename[:-5]
                    histories[model_path] = json.load(f)
                    
        return jsonify(histories)
        
    except Exception as e:
        print(f"Error getting chat history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/api/chat-history/<agent_id>', methods=['POST'])
def save_chat_history(agent_id):
    """Save chat history for a specific agent"""
    try:
        data = request.get_json()
        model_path = data.get('model_path')
        messages = data.get('messages', [])
        
        if not model_path:
            return jsonify({'error': 'Model path is required'}), 400
            
        history_path = get_agent_folder(agent_id, 'chat_history')
        if not os.path.exists(history_path):
            os.makedirs(history_path)
            
        history_file = os.path.join(history_path, f"{model_path}.json")
        with open(history_file, 'w') as f:
            json.dump(messages, f)
            
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"Error saving chat history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/chat-initialize', methods=['GET'])
def chat_initialize():
    """Initialize chat with safe memory cleanup"""
    try:
        global rag_chat
        
        if rag_chat is not None:
            rag_chat = None
            
        try:
            if RAGChat:
                RAGChat.reset_instance()
                print("Called RAGChat.reset_instance()")
        except Exception as reset_error:
            print(f"Warning: RAGChat.reset_instance() failed: {reset_error}")
        
        try:
            if clear_memory:
                clear_memory()
                print("Called clear_memory()")
        except Exception as memory_error:
            print(f"Warning: clear_memory() failed: {memory_error}")
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("PyTorch CUDA cache cleared")
        except Exception as cuda_error:
            print(f"Warning: CUDA cleanup failed: {cuda_error}")
        
        try:
            import gc
            gc.collect()
            print("Garbage collection completed")
        except Exception as gc_error:
            print(f"Warning: Garbage collection failed: {gc_error}")
            
        print("Memory cleanup completed safely")
        
        return jsonify({"status": "Chat Bot Initialized"}), 200
        
    except Exception as e:
        print(f"ERROR in chat_initialize: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@chat_bp.route('/chat-initialize-sdk', methods=['POST'])
def chat_initialize_sdk():
    """Initialize a chat session with the specified agent."""
    print("=== ENTERING chat_initialize_sdk ===")
    print(f"Request method: {request.method}")
    print(f"Request URL: {request.url}")
    print(f"Request headers: {dict(request.headers)}")
    
    try:
        global rag_chat
        
        print("Step 1: Getting JSON data...")
        data = request.get_json()
        print(f"Received data: {data}")
        
        if not data:
            print("ERROR: No data provided")
            return jsonify({'error': 'No data provided'}), 400
            
        agent = data.get('agent')
        model_name = data.get('model_name', 'unsloth/Qwen3-1.7B-bnb-4bit')
        
        print(f"Step 2: Agent: {agent}, Model: {model_name}")
        
        if not agent:
            print("ERROR: No agent provided")
            return jsonify({'error': 'Agent type is required'}), 400
            
        # Validate agent type
        valid_agents = ['general', 'tech', 'legal', 'finance', 'meetings', 'research', 'image-generator']
        if agent not in valid_agents:
            print(f"ERROR: Invalid agent: {agent}")
            return jsonify({'error': f'Invalid agent type. Must be one of: {valid_agents}'}), 400
        
        print(f"Step 3: Starting memory cleanup for agent: {agent}")
        
        # MINIMAL memory cleanup - remove ALL potentially problematic calls
        try:
            print("Step 3a: Clearing rag_chat variable...")
            if rag_chat is not None:
                rag_chat = None
                print("✓ rag_chat cleared")
            
            print("Step 3b: Memory cleanup completed")
            
        except Exception as cleanup_error:
            print(f"WARNING: Cleanup error (continuing anyway): {cleanup_error}")
        
        print("Step 4: Preparing response...")
        
        # For image generator, we don't need RAG chat initialization
        if agent == 'image-generator':
            print("Step 4a: Returning image generator response")
            return jsonify({
                'status': 'success',
                'message': f'Image generator agent initialized',
                'agent': agent,
                'model': 'FLUX.1-schnell'
            })
        
        # For other agents
        print("Step 4b: Returning general agent response")
        response = {
            'status': 'success',
            'message': f'{agent.title()} agent initialized successfully',
            'agent': agent,
            'model': model_name
        }
        print(f"Final response: {response}")
        return jsonify(response)
        
    except Exception as e:
        print(f"=== CRITICAL EXCEPTION in chat_initialize_sdk ===")
        print(f"Exception type: {type(e)}")
        print(f"Exception message: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=== END EXCEPTION ===")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@chat_bp.route('/chat', methods=['POST'])
def chat():
    global rag_chat, _CHAT_HISTORY, _CURRENT_AGENT
    if 'rag_chat' not in globals():
        rag_chat = None
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400
    user_input = data['message']
    agent = data['agent']
    model = data['model']
    system_command = data.get('system_command')
    include_history = data.get('include_history', True)
    web_search_enabled = data.get('web_search', False)
    web_context = None
    web_search_instruction = ""
    if web_search_enabled:
        try:
            rag_results = websearch_rag_pipeline(user_input)
            if rag_results:
                web_context = "\n\n".join([
                    f"Source: {item['title']}\nURL: {item['url']}\nContent: {item['content'][:1000]}" for item in rag_results
                ])
                web_search_instruction = (
                    "\n\n[Web Search Results] section is provided below. Use these results to answer the user's question. "
                    "Cite sources (by title or URL) in your answer where relevant. If the web results do not answer the question, say so."
                )
        except Exception as e:
            print(f"[WebSearch] Failed: {e}")
            web_context = None
            web_search_instruction = ""
    def generate():
        try:
            global rag_chat
            print("\nUser:", user_input)
            print("\nGenerating response...")
            yield 'data: {"status": "start"}\n\n'
            effective_input = user_input
            system_prompt = None
            if web_context:
                effective_input = f"[Web Search Results]\n{web_context}\n\n[User Question]\n{user_input}"
                system_prompt = (
                    "You are a knowledgeable and helpful AI assistant. "
                    "You have access to recent web search results. "
                    "Use the [Web Search Results] section to answer the user's question. "
                    "Cite sources (by title or URL) in your answer where relevant. "
                    "If the web results do not answer the question, say so. "
                    "Be concise, accurate, and professional."
                )
            if rag_chat is None:
                try:
                    fallback_model = load_default_chat_model(model)
                    fallback_model.check_agent_change(agent)
                    full_response = ""
                    for chunk, tokens_per_second in fallback_model.generate_response_stream(
                        effective_input,
                        agent,
                        include_history=include_history,
                        system_prompt=system_prompt if system_prompt else None
                    ):
                        full_response += chunk
                        print(chunk, end='', flush=True)
                        response_data = {
                            "response": chunk,
                            "tokens_per_second": round(tokens_per_second, 2)
                        }
                        yield f'data: {json.dumps(response_data)}\n\n'
                except Exception as e:
                    print(f"\nError in fallback model: {str(e)}")
                    error_data = {"error": str(e), "status": "error"}
                    yield f'data: {json.dumps(error_data)}\n\n'
                    return
            else:
                full_response = ""
                for chunk, tokens_per_second in rag_chat.generate_response_stream(
                    effective_input,
                    agent,
                    include_history=include_history,
                    system_prompt=system_prompt if system_prompt else None
                ):
                    full_response += chunk
                    print(chunk, end='', flush=True)
                    response_data = {
                        "response": chunk,
                        "tokens_per_second": round(tokens_per_second, 2)
                    }
                    yield f'data: {json.dumps(response_data)}\n\n'
            print("\n\nFull response complete.\n")
            yield f'data: {json.dumps({"response": full_response, "replace": True})}\n\n'
            yield 'data: {"status": "complete"}\n\n'
        except Exception as e:
            error_msg = f"\nError generating response: {str(e)}"
            print(error_msg)
            error_data = {"error": str(e), "status": "error"}
            yield f'data: {json.dumps(error_data)}\n\n'
    return Response(generate(), mimetype='text/event-stream')

@chat_bp.route('/chat/history', methods=['GET'])
def get_chat_history():
    """Get general chat history"""
    global rag_chat
    if rag_chat is None:
        return jsonify({"error": "Chat model not initialized"}), 400
    
    return jsonify({
        "history": rag_chat.chat_history.history
    })

@chat_bp.route('/test-sdk', methods=['POST'])
def test_sdk():
    """Simple test endpoint for SDK debugging"""
    print("=== ENTERING test_sdk ===")
    print(f"Request method: {request.method}")
    print(f"Request URL: {request.url}")
    
    try:
        print("Getting JSON data...")
        data = request.get_json()
        print(f"Test endpoint received: {data}")
        
        response = {
            'status': 'success',
            'message': 'Test endpoint working',
            'received_data': data,
            'timestamp': datetime.now().isoformat()
        }
        print(f"Returning response: {response}")
        return jsonify(response)
        
    except Exception as e:
        print(f"=== EXCEPTION in test_sdk ===")
        print(f"Exception type: {type(e)}")
        print(f"Exception message: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=== END EXCEPTION ===")
        return jsonify({'error': str(e)}), 500
