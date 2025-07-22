"""
Advanced Vision Chat routes module for Decompute Windows backend
Handles visual document processing, knowledge base management, and multi-modal chat
"""
#adde the vison llm processing of the image
import os
import sys
from pathlib import Path

# Load environment variables from web research pipeline .env file
def load_web_search_env():
    """Load environment variables from web research pipeline .env file"""
    try:
        # Find the web research pipeline directory
        current_dir = Path(__file__).parent
        web_research_dir = current_dir.parent.parent.parent / "web_research_pipeline"
        env_file = web_research_dir / ".env"
        
        if env_file.exists():
            print(f"Loading environment from: {env_file}")
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
            print("Environment variables loaded from web research pipeline .env file")
        else:
            print(f"Web research pipeline .env file not found at: {env_file}")
    except Exception as e:
        print(f"Error loading web research environment: {e}")

# Load environment variables
load_web_search_env()

from flask import Blueprint, request, jsonify, Response, stream_with_context
import torch
import json
import gc
import threading
import tempfile
import shutil
import uuid
import hashlib
from datetime import datetime
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# Import from core modules
from core.config import (
    BASE_UPLOAD_FOLDER, ALLOWED_EXTENSIONS, UPLOAD_FOLDER,
    SPECIAL_CHARS
)

# Import artifact processing
try:
    from routes.artifact_processing import info_preprocess, VisionLLMProcessor
except ImportError:
    info_preprocess = None
    VisionLLMProcessor = None

# Import RAG functionality
try:
    from rag_chat import RAGChat, clear_memory
except ImportError:
    RAGChat = None
    clear_memory = None

# Import embedding and indexing libraries
try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError:
    faiss = None
    np = None
    SentenceTransformer = None

# Import web search functionality
try:
    from web_research_pipeline.llm_web_search import (
        get_web_search_manager, 
        search_with_kb_context,
        search_web_for_query,
        WebSearchResult
    )
    WEB_SEARCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Web search not available: {e}")
    WEB_SEARCH_AVAILABLE = False
    get_web_search_manager = None
    search_with_kb_context = None
    search_web_for_query = None
    WebSearchResult = None

# Create blueprint
vision_chat_bp = Blueprint('vision_chat', __name__)

# Global variables for vision chat functionality
vision_rag_chat = None
_VISION_CURRENT_MODEL = None
_VISION_CURRENT_TOKENIZER = None
_VISION_CHAT_HISTORY = []
_VISION_CURRENT_AGENT = None
_VISION_KNOWLEDGE_BASES = {}
_VISION_LLM_PROCESSOR = None  # Global vision processor instance

def initialize_vision_processor():
    """
    Initialize the global VisionLLMProcessor instance for image processing.
    This should be called during vision chat initialization.
    """
    global _VISION_LLM_PROCESSOR
    
    try:
        if VisionLLMProcessor is None:
            print("⚠️  VisionLLMProcessor not available - using fallback image processing")
            return False
        
        if _VISION_LLM_PROCESSOR is None:
            print("🚀 Initializing Vision LLM Processor...")
            _VISION_LLM_PROCESSOR = VisionLLMProcessor()
            
            if _VISION_LLM_PROCESSOR.model is not None:
                print("✅ Vision LLM Processor initialized successfully")
                return True
            else:
                print("❌ Vision LLM Processor initialization failed - model not loaded")
                _VISION_LLM_PROCESSOR = None
                return False
        else:
            print("✅ Vision LLM Processor already initialized")
            return True
            
    except Exception as e:
        print(f"❌ Error initializing Vision LLM Processor: {e}")
        _VISION_LLM_PROCESSOR = None
        return False

def get_vision_processor():
    """
    Get the global VisionLLMProcessor instance.
    Returns None if not initialized.
    """
    global _VISION_LLM_PROCESSOR
    return _VISION_LLM_PROCESSOR

def clear_vision_processor():
    """
    Clear the global VisionLLMProcessor instance to free GPU memory.
    """
    global _VISION_LLM_PROCESSOR
    
    try:
        if _VISION_LLM_PROCESSOR is not None:
            print("🔄 Clearing Vision LLM Processor...")
            
            # Clear GPU cache if using CUDA
            if hasattr(_VISION_LLM_PROCESSOR, 'device') and _VISION_LLM_PROCESSOR.device == "cuda":
                try:
                    import torch
                    torch.cuda.empty_cache()
                    print("✅ GPU cache cleared")
                except Exception as e:
                    print(f"⚠️  Could not clear GPU cache: {e}")
            
            _VISION_LLM_PROCESSOR = None
            print("✅ Vision LLM Processor cleared")
            
    except Exception as e:
        print(f"❌ Error clearing Vision LLM Processor: {e}")
        _VISION_LLM_PROCESSOR = None

class KnowledgeBaseManager:
    """
    Manages knowledge bases for vision chat sessions with embedding-based retrieval
    """
    
    def __init__(self, session_id, agent_id):
        self.session_id = session_id
        self.agent_id = agent_id
        self.base_path = os.path.join(BASE_UPLOAD_FOLDER, 'knowledge_bases', session_id)
        self.documents_path = os.path.join(self.base_path, 'documents')
        self.images_path = os.path.join(self.base_path, 'images')
        self.processed_path = os.path.join(self.base_path, 'processed')
        self.index_path = os.path.join(self.base_path, 'index')
        self.metadata_path = os.path.join(self.base_path, 'metadata.json')
        
        # Embedding and indexing components
        self.embedding_model = None
        self.faiss_index = None
        self.embedding_dimension = 768
        self.chunk_mapping = {}
        self.embedding_metadata = {}
        
        # Create directory structure
        self._create_directory_structure()
        
        # Load or create metadata
        self.metadata = self._load_metadata()
        
        # Initialize embedding model and index
        self._initialize_embedding_system()
        
        # Use global vision processor for image processing
        self.vision_processor = get_vision_processor()
        
        # Initialize artifact processor as fallback
        self.artifact_processor = None
        if info_preprocess:
            try:
                self.artifact_processor = info_preprocess()
            except Exception as e:
                print(f"Warning: Could not initialize artifact processor: {e}")
    
    def _create_directory_structure(self):
        """Create the knowledge base directory structure"""
        directories = [
            self.base_path,
            self.documents_path,
            self.images_path,
            self.processed_path,
            self.index_path
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _initialize_embedding_system(self):
        """Initialize embedding model and FAISS index"""
        try:
            if SentenceTransformer is None:
                print("Warning: SentenceTransformer not available. Using fallback retrieval.")
                return
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer("hkunlp/instructor-large")
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
            
            # Initialize FAISS index
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dimension)
            
            # Load existing index if available
            self._load_existing_index()
            
            print(f"Embedding system initialized with dimension {self.embedding_dimension}")
            
        except Exception as e:
            print(f"Error initializing embedding system: {e}")
            self.embedding_model = None
            self.faiss_index = None
    
    def _load_existing_index(self):
        """Load existing FAISS index and chunk mapping"""
        try:
            index_file = os.path.join(self.index_path, "faiss_index.faiss")
            mapping_file = os.path.join(self.index_path, "chunk_mapping.json")
            metadata_file = os.path.join(self.index_path, "embedding_metadata.json")
            
            if (os.path.exists(index_file) and 
                os.path.exists(mapping_file) and 
                os.path.exists(metadata_file)):
                
                # Load FAISS index
                self.faiss_index = faiss.read_index(index_file)
                
                # Load chunk mapping
                with open(mapping_file, 'r') as f:
                    self.chunk_mapping = json.load(f)
                
                # Load embedding metadata
                with open(metadata_file, 'r') as f:
                    self.embedding_metadata = json.load(f)
                
                print(f"Loaded existing index with {self.faiss_index.ntotal} vectors")
                
        except Exception as e:
            print(f"Error loading existing index: {e}")
            # Reset to empty state
            if self.faiss_index is None:
                self.faiss_index = faiss.IndexFlatL2(self.embedding_dimension)
            self.chunk_mapping = {}
            self.embedding_metadata = {}
    
    def _save_index(self):
        """Save FAISS index and metadata"""
        try:
            if self.faiss_index is None:
                return
            
            # Save FAISS index
            index_file = os.path.join(self.index_path, "faiss_index.faiss")
            faiss.write_index(self.faiss_index, index_file)
            
            # Save chunk mapping
            mapping_file = os.path.join(self.index_path, "chunk_mapping.json")
            with open(mapping_file, 'w') as f:
                json.dump(self.chunk_mapping, f, indent=2)
            
            # Save embedding metadata
            metadata_file = os.path.join(self.index_path, "embedding_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(self.embedding_metadata, f, indent=2)
            
            print(f"Saved index with {self.faiss_index.ntotal} vectors")
            
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def _create_chunks(self, text, chunk_size=1000, chunk_overlap=200):
        """Create text chunks for embedding"""
        if not text.strip():
            return []
        
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def _add_content_to_index(self, content_id, content_type, text_content, metadata=None):
        """Add content to the embedding index"""
        try:
            if self.embedding_model is None or self.faiss_index is None:
                return
            
            # Create chunks from text content
            chunks = self._create_chunks(text_content)
            
            if not chunks:
                return
            
            # Generate embeddings for chunks
            embeddings = self.embedding_model.encode(chunks, normalize_embeddings=True)
            
            # Add to FAISS index
            start_idx = self.faiss_index.ntotal
            self.faiss_index.add(embeddings)
            
            # Update chunk mapping and metadata
            for i, chunk in enumerate(chunks):
                chunk_id = start_idx + i
                self.chunk_mapping[str(chunk_id)] = chunk
                
                self.embedding_metadata[str(chunk_id)] = {
                    'content_id': content_id,
                    'content_type': content_type,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'metadata': metadata or {}
                }
            
            # Save updated index
            self._save_index()
            
            print(f"Added {len(chunks)} chunks for {content_type} {content_id}")
            
        except Exception as e:
            print(f"Error adding content to index: {e}")
    
    def _load_metadata(self):
        """Load or create metadata file"""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading metadata: {e}")
        
        # Create new metadata
        metadata = {
            'session_id': self.session_id,
            'agent_id': self.agent_id,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'documents': {},
            'images': {},
            'processed_content': {},
            'index_info': {
                'total_chunks': 0,
                'last_indexed': None,
                'index_version': '2.0',
                'embedding_model': 'hkunlp/instructor-large'
            }
        }
        self._save_metadata(metadata)
        return metadata
    
    def _save_metadata(self, metadata=None):
        """Save metadata to file"""
        if metadata is None:
            metadata = self.metadata
        metadata['last_updated'] = datetime.now().isoformat()
        
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def add_document(self, file_path, file_type='pdf'):
        """
        Add a document to the knowledge base and process it
        """
        try:
            # Generate unique document ID
            doc_id = str(uuid.uuid4())
            doc_name = os.path.basename(file_path)
            
            # Copy document to knowledge base
            dest_path = os.path.join(self.documents_path, f"{doc_id}_{doc_name}")
            shutil.copy2(file_path, dest_path)
            
            # Process document based on type
            if file_type.lower() == 'pdf':
                processed_content = self._process_pdf_document(dest_path, doc_id)
            else:
                processed_content = self._process_general_document(dest_path, doc_id)
            
            # Update metadata
            self.metadata['documents'][doc_id] = {
                'name': doc_name,
                'path': dest_path,
                'type': file_type,
                'added_at': datetime.now().isoformat(),
                'processed': True,
                'content_summary': processed_content.get('summary', ''),
                'pages': processed_content.get('pages', 0)
            }
            
            # Store processed content
            processed_file = os.path.join(self.processed_path, f"{doc_id}_processed.json")
            with open(processed_file, 'w') as f:
                json.dump(processed_content, f, indent=2)
            
            self.metadata['processed_content'][doc_id] = {
                'type': 'document',
                'file': processed_file,
                'processed_at': datetime.now().isoformat()
            }
            
            # Add to embedding index
            self._add_document_to_index(doc_id, processed_content)
            
            self._save_metadata()
            
            return {
                'doc_id': doc_id,
                'name': doc_name,
                'processed': True,
                'content': processed_content
            }
            
        except Exception as e:
            print(f"Error adding document: {e}")
            return {'error': str(e)}
    
    def _add_document_to_index(self, doc_id, processed_content):
        """Add document content to embedding index"""
        try:
            if processed_content.get('type') == 'pdf':
                # For PDFs, add each page's content
                for page_num, page_data in processed_content.get('pages', {}).items():
                    # Combine text and enhanced content
                    page_text = page_data.get('text', '') + ' ' + page_data.get('enhanced_content', '')
                    
                    # Add page content to index
                    self._add_content_to_index(
                        content_id=doc_id,
                        content_type='pdf_page',
                        text_content=page_text,
                        metadata={
                            'page_number': page_num,
                            'document_name': processed_content.get('doc_id', ''),
                            'has_images': len(page_data.get('images', [])) > 0
                        }
                    )
                    
                    # Add image descriptions if available
                    for img in page_data.get('images', []):
                        img_text = f"Image {img.get('index', '')}: {img.get('description', '')}"
                        self._add_content_to_index(
                            content_id=f"{doc_id}_img_{img.get('index', '')}",
                            content_type='pdf_image',
                            text_content=img_text,
                            metadata={
                                'page_number': page_num,
                                'document_id': doc_id,
                                'image_path': img.get('path', ''),
                                'image_index': img.get('index', '')
                            }
                        )
            else:
                # For general documents, add the content directly
                content = processed_content.get('content', '')
                self._add_content_to_index(
                    content_id=doc_id,
                    content_type='document',
                    text_content=content,
                    metadata={
                        'document_name': processed_content.get('doc_id', ''),
                        'document_type': processed_content.get('type', '')
                    }
                )
                
        except Exception as e:
            print(f"Error adding document to index: {e}")
    
    def add_image(self, file_path):
        """
        Add an image to the knowledge base and process it
        """
        try:
            print(f"DEBUG: add_image called with file_path: {file_path}")
            print(f"DEBUG: File exists: {os.path.exists(file_path)}")
            
            # Generate unique image ID
            img_id = str(uuid.uuid4())
            img_name = os.path.basename(file_path)
            
            print(f"DEBUG: Generated img_id: {img_id}")
            print(f"DEBUG: Image name: {img_name}")
            print(f"DEBUG: Images path: {self.images_path}")
            
            # Copy image to knowledge base
            dest_path = os.path.join(self.images_path, f"{img_id}_{img_name}")
            print(f"DEBUG: Destination path: {dest_path}")
            
            shutil.copy2(file_path, dest_path)
            print(f"DEBUG: Image copied successfully to: {dest_path}")
            print(f"DEBUG: Destination file exists: {os.path.exists(dest_path)}")
            print(f"DEBUG: Destination file size: {os.path.getsize(dest_path) if os.path.exists(dest_path) else 'N/A'} bytes")
            
            # Process image using Vision LLM processor or artifact processor
            processed_content = self._process_image(dest_path, img_id)
            print(f"DEBUG: Processed content: {processed_content}")
            
            # Update metadata
            self.metadata['images'][img_id] = {
                'name': img_name,
                'path': dest_path,
                'added_at': datetime.now().isoformat(),
                'processed': True,
                'description': processed_content.get('description', '')
            }
            
            # Store processed content
            processed_file = os.path.join(self.processed_path, f"{img_id}_processed.json")
            with open(processed_file, 'w') as f:
                json.dump(processed_content, f, indent=2)
            
            self.metadata['processed_content'][img_id] = {
                'type': 'image',
                'file': processed_file,
                'processed_at': datetime.now().isoformat()
            }
            
            # Add to embedding index
            self._add_image_to_index(img_id, processed_content)
            
            self._save_metadata()
            print(f"DEBUG: Image successfully added to knowledge base")
            
            return {
                'img_id': img_id,
                'name': img_name,
                'processed': True,
                'content': processed_content
            }
            
        except Exception as e:
            print(f"DEBUG: Error adding image: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def _add_image_to_index(self, img_id, processed_content):
        """Add image content to embedding index"""
        try:
            # Add image description to index
            description = processed_content.get('description', '')
            if description:
                self._add_content_to_index(
                    content_id=img_id,
                    content_type='image',
                    text_content=description,
                    metadata={
                        'image_name': processed_content.get('img_id', ''),
                        'image_path': processed_content.get('path', ''),
                        'content_type': 'image_description'
                    }
                )
                
        except Exception as e:
            print(f"Error adding image to index: {e}")
    
    def get_relevant_content(self, query, max_results=5):
        """
        Get relevant content from knowledge base using embedding-based retrieval
        """
        try:
            if self.embedding_model is None or self.faiss_index is None:
                # Fallback to keyword-based search
                return self._fallback_keyword_search(query, max_results)
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
            
            # Search FAISS index
            k = min(max_results * 3, self.faiss_index.ntotal)  # Get more results for filtering
            if k <= 0:
                return []
            
            distances, indices = self.faiss_index.search(query_embedding, k)
            
            # Process and score results
            scored_results = []
            for dist, idx in zip(distances[0], indices[0]):
                chunk_id = str(idx)
                chunk_text = self.chunk_mapping.get(chunk_id, '')
                metadata = self.embedding_metadata.get(chunk_id, {})
                
                if not chunk_text:
                    continue
                
                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(query, chunk_text, dist, metadata)
                
                if relevance_score > 0.3:  # Minimum relevance threshold
                    scored_results.append({
                        'chunk_id': chunk_id,
                        'content': chunk_text,
                        'metadata': metadata,
                        'relevance_score': relevance_score,
                        'distance': float(dist)
                    })
            
            # Sort by relevance score and take top results
            scored_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Group by content source and deduplicate
            grouped_results = self._group_and_deduplicate_results(scored_results, max_results)
            
            return grouped_results
            
        except Exception as e:
            print(f"Error in embedding-based retrieval: {e}")
            return self._fallback_keyword_search(query, max_results)
    
    def _calculate_relevance_score(self, query, chunk_text, distance, metadata):
        """Calculate relevance score using multiple factors"""
        try:
            # Base score from embedding distance (inverse)
            distance_score = 1.0 / (1.0 + distance)
            
            # Keyword overlap score
            query_words = set(query.lower().split())
            chunk_words = set(chunk_text.lower().split())
            keyword_overlap = len(query_words.intersection(chunk_words)) / len(query_words) if query_words else 0
            
            # Content type bonus
            content_type = metadata.get('content_type', '')
            type_bonus = 1.0
            if content_type == 'pdf_page':
                type_bonus = 1.2  # Slightly prefer document pages
            elif content_type == 'image':
                type_bonus = 1.1  # Slightly prefer images for visual queries
            
            # Length penalty (prefer concise but informative chunks)
            length_penalty = min(1.0, 200 / max(len(chunk_text), 1))
            
            # Combined score
            final_score = (
                0.6 * distance_score +
                0.3 * keyword_overlap +
                0.1 * length_penalty
            ) * type_bonus
            
            return min(1.0, final_score)
            
        except Exception as e:
            print(f"Error calculating relevance score: {e}")
            return 0.5
    
    def _group_and_deduplicate_results(self, scored_results, max_results):
        """Group results by content source and remove duplicates"""
        try:
            grouped = {}
            
            for result in scored_results:
                content_id = result['metadata'].get('content_id', 'unknown')
                
                if content_id not in grouped:
                    grouped[content_id] = {
                        'content_id': content_id,
                        'content_type': result['metadata'].get('content_type', ''),
                        'best_chunk': result['content'],
                        'best_score': result['relevance_score'],
                        'metadata': result['metadata'],
                        'all_chunks': [result['content']],
                        'total_score': result['relevance_score']
                    }
                else:
                    # Update with better chunk if found
                    if result['relevance_score'] > grouped[content_id]['best_score']:
                        grouped[content_id]['best_chunk'] = result['content']
                        grouped[content_id]['best_score'] = result['relevance_score']
                        grouped[content_id]['metadata'] = result['metadata']
                    
                    grouped[content_id]['all_chunks'].append(result['content'])
                    grouped[content_id]['total_score'] += result['relevance_score']
            
            # Sort by best score and take top results
            sorted_results = sorted(grouped.values(), key=lambda x: x['best_score'], reverse=True)
            
            return sorted_results[:max_results]
            
        except Exception as e:
            print(f"Error grouping results: {e}")
            return scored_results[:max_results]
    
    def _fallback_keyword_search(self, query, max_results=5):
        """Fallback keyword-based search when embeddings are not available"""
        relevant_content = []
        
        # Search through processed content
        for content_id, content_info in self.metadata.get('processed_content', {}).items():
            try:
                processed_file = content_info.get('file')
                if os.path.exists(processed_file):
                    with open(processed_file, 'r') as f:
                        content_data = json.load(f)
                    
                    # Simple keyword matching
                    query_lower = query.lower()
                    content_text = ""
                    
                    if content_info['type'] == 'document':
                        if 'content' in content_data:
                            content_text = content_data['content']
                        elif 'pages' in content_data:
                            for page_data in content_data['pages'].values():
                                content_text += page_data.get('text', '') + " "
                                content_text += page_data.get('enhanced_content', '') + " "
                    elif content_info['type'] == 'image':
                        content_text = content_data.get('description', '')
                    
                    # Check if query terms are in content
                    if any(term in content_text.lower() for term in query_lower.split()):
                        relevant_content.append({
                            'content_id': content_id,
                            'content_type': content_info['type'],
                            'content': content_data,
                            'relevance_score': 0.8  # Placeholder score
                        })
                        
                        if len(relevant_content) >= max_results:
                            break
                            
            except Exception as e:
                print(f"Error retrieving content {content_id}: {e}")
                continue
        
        return relevant_content

    def _process_pdf_document(self, pdf_path, doc_id):
        """
        Process PDF document with visual content extraction
        """
        try:
            processed_content = {
                'doc_id': doc_id,
                'type': 'pdf',
                'pages': {},
                'images': {},
                'summary': '',
                'processed_at': datetime.now().isoformat()
            }
            
            # Open PDF
            pdf_document = fitz.open(pdf_path)
            total_pages = len(pdf_document)
            
            # Limit processing to prevent infinite loops
            max_pages = min(total_pages, 50)
            
            for page_num in range(max_pages):
                try:
                    page = pdf_document[page_num]
                    
                    # Extract text
                    text_content = page.get_text()
                    
                    # Extract images from page
                    image_list = page.get_images()
                    page_images = []
                    
                    for img_index, img in enumerate(image_list):
                        try:
                            # Get image data
                            xref = img[0]
                            base_image = pdf_document.extract_image(xref)
                            image_bytes = base_image["image"]
                            
                            # Save image
                            img_filename = f"{doc_id}_page_{page_num + 1}_img_{img_index + 1}.png"
                            img_path = os.path.join(self.images_path, img_filename)
                            
                            with open(img_path, "wb") as img_file:
                                img_file.write(image_bytes)
                            
                            # Process image if artifact processor is available
                            if self.artifact_processor:
                                try:
                                    img_content = self.artifact_processor.get_image_description(img_path)
                                    page_images.append({
                                        'path': img_path,
                                        'description': img_content,
                                        'index': img_index + 1
                                    })
                                except Exception as img_e:
                                    print(f"Error processing image on page {page_num + 1}: {img_e}")
                                    page_images.append({
                                        'path': img_path,
                                        'description': f"Image {img_index + 1} from page {page_num + 1}",
                                        'index': img_index + 1
                                    })
                            
                        except Exception as e:
                            print(f"Error extracting image {img_index} from page {page_num + 1}: {e}")
                            continue
                    
                    # Process page content with artifact processor if available
                    if self.artifact_processor:
                        try:
                            # Convert page to image for OCR
                            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                            page_img_path = os.path.join(self.images_path, f"{doc_id}_page_{page_num + 1}.png")
                            pix.save(page_img_path)
                            
                            # Get enhanced description
                            enhanced_content = self.artifact_processor.get_image_description(page_img_path)
                            
                            processed_content['pages'][page_num + 1] = {
                                'text': text_content,
                                'enhanced_content': enhanced_content,
                                'images': page_images,
                                'page_number': page_num + 1
                            }
                            
                            # Clean up temporary page image
                            if os.path.exists(page_img_path):
                                os.remove(page_img_path)
                                
                        except Exception as e:
                            print(f"Error processing page {page_num + 1} with artifact processor: {e}")
                            processed_content['pages'][page_num + 1] = {
                                'text': text_content,
                                'enhanced_content': f"Page {page_num + 1} content",
                                'images': page_images,
                                'page_number': page_num + 1
                            }
                    else:
                        processed_content['pages'][page_num + 1] = {
                            'text': text_content,
                            'enhanced_content': f"Page {page_num + 1} content",
                            'images': page_images,
                            'page_number': page_num + 1
                        }
                    
                except Exception as e:
                    print(f"Error processing page {page_num + 1}: {e}")
                    processed_content['pages'][page_num + 1] = {
                        'text': f"Error processing page: {str(e)}",
                        'enhanced_content': f"Error processing page {page_num + 1}",
                        'images': [],
                        'page_number': page_num + 1
                    }
            
            pdf_document.close()
            
            # Create summary
            total_text = ""
            for page_data in processed_content['pages'].values():
                total_text += page_data.get('text', '') + " "
            
            processed_content['summary'] = total_text[:500] + "..." if len(total_text) > 500 else total_text
            processed_content['pages'] = len(processed_content['pages'])
            
            return processed_content
            
        except Exception as e:
            print(f"Error processing PDF document: {e}")
            return {
                'doc_id': doc_id,
                'type': 'pdf',
                'error': str(e),
                'pages': 0,
                'summary': f"Error processing PDF: {str(e)}"
            }
    
    def _process_general_document(self, doc_path, doc_id):
        """
        Process general document types
        """
        try:
            # For now, treat as text document
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                'doc_id': doc_id,
                'type': 'document',
                'content': content,
                'summary': content[:500] + "..." if len(content) > 500 else content,
                'pages': 1,
                'processed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error processing general document: {e}")
            return {
                'doc_id': doc_id,
                'type': 'document',
                'error': str(e),
                'summary': f"Error processing document: {str(e)}"
            }
    
    def _process_image(self, img_path, img_id):
        """
        Process image using Vision LLM processor or artifact processor
        """
        try:
            description = ""
            
            # Try using the global Vision LLM processor first
            if self.vision_processor and self.vision_processor.model is not None:
                print(f"🤖 Using Vision LLM Processor for image: {img_path}")
                description = self.vision_processor.get_vision_description(img_path)
            elif self.artifact_processor:
                print(f"📄 Using Artifact Processor for image: {img_path}")
                description = self.artifact_processor.get_image_description(img_path)
            else:
                # Fallback description
                print(f"⚠️  No image processor available, using fallback for: {img_path}")
                description = f"Image: {os.path.basename(img_path)} - No image processor available"
            
            return {
                'img_id': img_id,
                'type': 'image',
                'description': description,
                'path': img_path,
                'processed_at': datetime.now().isoformat(),
                'processor_used': 'vision_llm' if (self.vision_processor and self.vision_processor.model is not None) else 'artifact_processor' if self.artifact_processor else 'fallback'
            }
            
        except Exception as e:
            print(f"❌ Error processing image {img_path}: {e}")
            return {
                'img_id': img_id,
                'type': 'image',
                'error': str(e),
                'description': f"Error processing image: {str(e)}",
                'processor_used': 'error'
            }
    
    def get_knowledge_base_summary(self):
        """
        Get summary of knowledge base content
        """
        # Update index info with current stats
        if self.faiss_index is not None:
            self.metadata['index_info']['total_chunks'] = self.faiss_index.ntotal
            self.metadata['index_info']['last_indexed'] = datetime.now().isoformat()
        
        return {
            'session_id': self.session_id,
            'agent_id': self.agent_id,
            'total_documents': len(self.metadata.get('documents', {})),
            'total_images': len(self.metadata.get('images', {})),
            'total_processed': len(self.metadata.get('processed_content', {})),
            'created_at': self.metadata.get('created_at'),
            'last_updated': self.metadata.get('last_updated'),
            'index_info': self.metadata.get('index_info', {})
        }

    def delete_document(self, doc_id):
        """
        Delete document and all associated content from knowledge base
        """
        try:
            if doc_id not in self.metadata.get('documents', {}):
                return {'error': f'Document {doc_id} not found'}
            
            doc_info = self.metadata['documents'][doc_id]
            doc_path = doc_info.get('path', '')
            
            # Remove from embedding index
            self._remove_content_from_index(doc_id)
            
            # Delete original document file
            if os.path.exists(doc_path):
                os.remove(doc_path)
                print(f"Deleted document file: {doc_path}")
            
            # Delete processed content file
            processed_file = self.metadata.get('processed_content', {}).get(doc_id, {}).get('file', '')
            if processed_file and os.path.exists(processed_file):
                os.remove(processed_file)
                print(f"Deleted processed file: {processed_file}")
            
            # Delete associated images from PDF extraction
            if doc_info.get('type') == 'pdf':
                self._delete_document_images(doc_id)
            
            # Remove from metadata
            if doc_id in self.metadata['documents']:
                del self.metadata['documents'][doc_id]
            
            if doc_id in self.metadata.get('processed_content', {}):
                del self.metadata['processed_content'][doc_id]
            
            # Save updated metadata
            self._save_metadata()
            
            # Rebuild index to ensure consistency
            self._rebuild_index_without_content(doc_id)
            
            return {
                'status': 'success',
                'message': f'Document {doc_id} deleted successfully',
                'deleted_doc_id': doc_id
            }
            
        except Exception as e:
            print(f"Error deleting document {doc_id}: {e}")
            return {'error': str(e)}

    def delete_image(self, img_id):
        """
        Delete image and all associated content from knowledge base
        """
        try:
            if img_id not in self.metadata.get('images', {}):
                return {'error': f'Image {img_id} not found'}
            
            img_info = self.metadata['images'][img_id]
            img_path = img_info.get('path', '')
            
            # Remove from embedding index
            self._remove_content_from_index(img_id)
            
            # Delete original image file
            if os.path.exists(img_path):
                os.remove(img_path)
                print(f"Deleted image file: {img_path}")
            
            # Delete processed content file
            processed_file = self.metadata.get('processed_content', {}).get(img_id, {}).get('file', '')
            if processed_file and os.path.exists(processed_file):
                os.remove(processed_file)
                print(f"Deleted processed file: {processed_file}")
            
            # Remove from metadata
            if img_id in self.metadata['images']:
                del self.metadata['images'][img_id]
            
            if img_id in self.metadata.get('processed_content', {}):
                del self.metadata['processed_content'][img_id]
            
            # Save updated metadata
            self._save_metadata()
            
            # Rebuild index to ensure consistency
            self._rebuild_index_without_content(img_id)
            
            return {
                'status': 'success',
                'message': f'Image {img_id} deleted successfully',
                'deleted_img_id': img_id
            }
            
        except Exception as e:
            print(f"Error deleting image {img_id}: {e}")
            return {'error': str(e)}

    def delete_page_from_document(self, doc_id, page_number):
        """
        Delete specific page from PDF document
        """
        try:
            if doc_id not in self.metadata.get('documents', {}):
                return {'error': f'Document {doc_id} not found'}
            
            doc_info = self.metadata['documents'][doc_id]
            if doc_info.get('type') != 'pdf':
                return {'error': 'Page deletion only supported for PDF documents'}
            
            # Get processed content
            processed_file = self.metadata.get('processed_content', {}).get(doc_id, {}).get('file', '')
            if not processed_file or not os.path.exists(processed_file):
                return {'error': 'Processed content not found'}
            
            with open(processed_file, 'r') as f:
                content_data = json.load(f)
            
            pages = content_data.get('pages', {})
            if str(page_number) not in pages:
                return {'error': f'Page {page_number} not found in document'}
            
            # Remove page from processed content
            del pages[str(page_number)]
            
            # Update page count
            content_data['pages'] = len(pages)
            
            # Save updated processed content
            with open(processed_file, 'w') as f:
                json.dump(content_data, f, indent=2)
            
            # Remove page chunks from embedding index
            self._remove_page_from_index(doc_id, page_number)
            
            # Update metadata
            self.metadata['documents'][doc_id]['pages'] = len(pages)
            self._save_metadata()
            
            return {
                'status': 'success',
                'message': f'Page {page_number} deleted from document {doc_id}',
                'remaining_pages': len(pages)
            }
            
        except Exception as e:
            print(f"Error deleting page {page_number} from document {doc_id}: {e}")
            return {'error': str(e)}

    def clear_knowledge_base(self):
        """
        Clear entire knowledge base
        """
        try:
            # Clear embedding index
            if self.faiss_index is not None:
                self.faiss_index = faiss.IndexFlatL2(self.embedding_dimension)
            self.chunk_mapping = {}
            self.embedding_metadata = {}
            
            # Delete all document files
            for doc_id, doc_info in self.metadata.get('documents', {}).items():
                doc_path = doc_info.get('path', '')
                if os.path.exists(doc_path):
                    os.remove(doc_path)
            
            # Delete all image files
            for img_id, img_info in self.metadata.get('images', {}).items():
                img_path = img_info.get('path', '')
                if os.path.exists(img_path):
                    os.remove(img_path)
            
            # Delete all processed content files
            for content_id, content_info in self.metadata.get('processed_content', {}).items():
                processed_file = content_info.get('file', '')
                if os.path.exists(processed_file):
                    os.remove(processed_file)
            
            # Clear metadata
            self.metadata['documents'] = {}
            self.metadata['images'] = {}
            self.metadata['processed_content'] = {}
            self.metadata['index_info']['total_chunks'] = 0
            self.metadata['index_info']['last_indexed'] = datetime.now().isoformat()
            
            # Save cleared metadata
            self._save_metadata()
            
            # Save empty index
            self._save_index()
            
            return {
                'status': 'success',
                'message': 'Knowledge base cleared successfully',
                'cleared_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error clearing knowledge base: {e}")
            return {'error': str(e)}

    def _remove_content_from_index(self, content_id):
        """
        Remove content from embedding index by rebuilding without the specified content
        """
        try:
            if self.faiss_index is None or self.embedding_model is None:
                return
            
            # Create new index without the content
            new_index = faiss.IndexFlatL2(self.embedding_dimension)
            new_chunk_mapping = {}
            new_embedding_metadata = {}
            
            # Rebuild index excluding the content to be deleted
            for chunk_id, chunk_text in self.chunk_mapping.items():
                chunk_metadata = self.embedding_metadata.get(chunk_id, {})
                chunk_content_id = chunk_metadata.get('content_id', '')
                
                # Skip chunks belonging to the content to be deleted
                if chunk_content_id == content_id:
                    continue
                
                # Add chunk to new index
                embedding = self.embedding_model.encode([chunk_text], normalize_embeddings=True)
                new_index.add(embedding)
                
                # Update mappings
                new_chunk_id = str(new_index.ntotal - 1)
                new_chunk_mapping[new_chunk_id] = chunk_text
                new_embedding_metadata[new_chunk_id] = chunk_metadata
            
            # Replace old index with new one
            self.faiss_index = new_index
            self.chunk_mapping = new_chunk_mapping
            self.embedding_metadata = new_embedding_metadata
            
            print(f"Removed content {content_id} from embedding index")
            
        except Exception as e:
            print(f"Error removing content from index: {e}")

    def _rebuild_index_without_content(self, content_id):
        """
        Rebuild FAISS index excluding specific content
        """
        try:
            if self.faiss_index is None or self.embedding_model is None:
                return
            
            # Create new index
            new_index = faiss.IndexFlatL2(self.embedding_dimension)
            new_chunk_mapping = {}
            new_embedding_metadata = {}
            
            # Rebuild from all processed content except the deleted one
            for content_info_id, content_info in self.metadata.get('processed_content', {}).items():
                if content_info_id == content_id:
                    continue
                
                processed_file = content_info.get('file')
                if not os.path.exists(processed_file):
                    continue
                
                try:
                    with open(processed_file, 'r') as f:
                        content_data = json.load(f)
                    
                    # Re-add content to index based on type
                    if content_info['type'] == 'document':
                        if content_data.get('type') == 'pdf':
                            # Re-add PDF pages
                            for page_num, page_data in content_data.get('pages', {}).items():
                                page_text = page_data.get('text', '') + ' ' + page_data.get('enhanced_content', '')
                                self._add_content_to_new_index(
                                    new_index, new_chunk_mapping, new_embedding_metadata,
                                    content_id=content_info_id,
                                    content_type='pdf_page',
                                    text_content=page_text,
                                    metadata={
                                        'page_number': page_num,
                                        'document_name': content_data.get('doc_id', ''),
                                        'has_images': len(page_data.get('images', [])) > 0
                                    }
                                )
                        else:
                            # Re-add general document
                            content = content_data.get('content', '')
                            self._add_content_to_new_index(
                                new_index, new_chunk_mapping, new_embedding_metadata,
                                content_id=content_info_id,
                                content_type='document',
                                text_content=content,
                                metadata={
                                    'document_name': content_data.get('doc_id', ''),
                                    'document_type': content_data.get('type', '')
                                }
                            )
                    
                    elif content_info['type'] == 'image':
                        # Re-add image description
                        description = content_data.get('description', '')
                        if description:
                            self._add_content_to_new_index(
                                new_index, new_chunk_mapping, new_embedding_metadata,
                                content_id=content_info_id,
                                content_type='image',
                                text_content=description,
                                metadata={
                                    'image_name': content_data.get('img_id', ''),
                                    'image_path': content_data.get('path', ''),
                                    'content_type': 'image_description'
                                }
                            )
                            
                except Exception as e:
                    print(f"Error rebuilding index for content {content_info_id}: {e}")
                    continue
            
            # Replace old index with new one
            self.faiss_index = new_index
            self.chunk_mapping = new_chunk_mapping
            self.embedding_metadata = new_embedding_metadata
            
            # Save the rebuilt index
            self._save_index()
            
            print(f"Rebuilt index without content {content_id}")
            
        except Exception as e:
            print(f"Error rebuilding index: {e}")

    def _add_content_to_new_index(self, new_index, new_chunk_mapping, new_embedding_metadata, 
                                 content_id, content_type, text_content, metadata=None):
        """
        Add content to new index during rebuild
        """
        try:
            # Create chunks from text content
            chunks = self._create_chunks(text_content)
            
            if not chunks:
                return
            
            # Generate embeddings for chunks
            embeddings = self.embedding_model.encode(chunks, normalize_embeddings=True)
            
            # Add to new index
            start_idx = new_index.ntotal
            new_index.add(embeddings)
            
            # Update new chunk mapping and metadata
            for i, chunk in enumerate(chunks):
                chunk_id = str(start_idx + i)
                new_chunk_mapping[chunk_id] = chunk
                
                new_embedding_metadata[chunk_id] = {
                    'content_id': content_id,
                    'content_type': content_type,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'metadata': metadata or {}
                }
                
        except Exception as e:
            print(f"Error adding content to new index: {e}")

    def _remove_page_from_index(self, doc_id, page_number):
        """
        Remove specific page chunks from embedding index
        """
        try:
            if self.faiss_index is None or self.embedding_model is None:
                return
            
            # Create new index without the page
            new_index = faiss.IndexFlatL2(self.embedding_dimension)
            new_chunk_mapping = {}
            new_embedding_metadata = {}
            
            # Rebuild index excluding the page to be deleted
            for chunk_id, chunk_text in self.chunk_mapping.items():
                chunk_metadata = self.embedding_metadata.get(chunk_id, {})
                chunk_content_id = chunk_metadata.get('content_id', '')
                chunk_page_number = chunk_metadata.get('metadata', {}).get('page_number')
                
                # Skip chunks belonging to the page to be deleted
                if (chunk_content_id == doc_id and 
                    chunk_metadata.get('content_type') == 'pdf_page' and
                    str(chunk_page_number) == str(page_number)):
                    continue
                
                # Add chunk to new index
                embedding = self.embedding_model.encode([chunk_text], normalize_embeddings=True)
                new_index.add(embedding)
                
                # Update mappings
                new_chunk_id = str(new_index.ntotal - 1)
                new_chunk_mapping[new_chunk_id] = chunk_text
                new_embedding_metadata[new_chunk_id] = chunk_metadata
            
            # Replace old index with new one
            self.faiss_index = new_index
            self.chunk_mapping = new_chunk_mapping
            self.embedding_metadata = new_embedding_metadata
            
            print(f"Removed page {page_number} from document {doc_id} in embedding index")
            
        except Exception as e:
            print(f"Error removing page from index: {e}")

    def _delete_document_images(self, doc_id):
        """
        Delete images extracted from a PDF document
        """
        try:
            # Find and delete images with document ID prefix
            for filename in os.listdir(self.images_path):
                if filename.startswith(f"{doc_id}_"):
                    img_path = os.path.join(self.images_path, filename)
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        print(f"Deleted extracted image: {img_path}")
                        
        except Exception as e:
            print(f"Error deleting document images: {e}")

    def bulk_delete_content(self, content_ids, content_type='document'):
        """
        Bulk delete multiple content items
        """
        try:
            results = []
            errors = []
            
            for content_id in content_ids:
                if content_type == 'document':
                    result = self.delete_document(content_id)
                elif content_type == 'image':
                    result = self.delete_image(content_id)
                else:
                    result = {'error': f'Invalid content type: {content_type}'}
                
                if 'error' in result:
                    errors.append({'content_id': content_id, 'error': result['error']})
                else:
                    results.append({'content_id': content_id, 'status': 'success'})
            
            return {
                'status': 'completed',
                'successful_deletions': len(results),
                'failed_deletions': len(errors),
                'results': results,
                'errors': errors
            }
            
        except Exception as e:
            print(f"Error in bulk delete: {e}")
            return {'error': str(e)}

    def get_deletion_preview(self, content_id, content_type='document'):
        """
        Get preview of what will be deleted
        """
        try:
            preview = {
                'content_id': content_id,
                'content_type': content_type,
                'files_to_delete': [],
                'index_chunks_to_remove': 0,
                'metadata_entries_to_remove': 0
            }
            
            if content_type == 'document':
                if content_id in self.metadata.get('documents', {}):
                    doc_info = self.metadata['documents'][content_id]
                    preview['files_to_delete'].append(doc_info.get('path', ''))
                    
                    # Count associated chunks
                    for chunk_meta in self.embedding_metadata.values():
                        if chunk_meta.get('content_id') == content_id:
                            preview['index_chunks_to_remove'] += 1
                    
                    # Count associated images if PDF
                    if doc_info.get('type') == 'pdf':
                        for filename in os.listdir(self.images_path):
                            if filename.startswith(f"{content_id}_"):
                                preview['files_to_delete'].append(os.path.join(self.images_path, filename))
                    
                    preview['metadata_entries_to_remove'] = 2  # documents + processed_content
                    
            elif content_type == 'image':
                if content_id in self.metadata.get('images', {}):
                    img_info = self.metadata['images'][content_id]
                    preview['files_to_delete'].append(img_info.get('path', ''))
                    
                    # Count associated chunks
                    for chunk_meta in self.embedding_metadata.values():
                        if chunk_meta.get('content_id') == content_id:
                            preview['index_chunks_to_remove'] += 1
                    
                    preview['metadata_entries_to_remove'] = 2  # images + processed_content
            
            return preview
            
        except Exception as e:
            print(f"Error getting deletion preview: {e}")
            return {'error': str(e)}

def get_knowledge_base(session_id, agent_id):
    """Get or create knowledge base for session"""
    if session_id not in _VISION_KNOWLEDGE_BASES:
        _VISION_KNOWLEDGE_BASES[session_id] = KnowledgeBaseManager(session_id, agent_id)
    return _VISION_KNOWLEDGE_BASES[session_id]



def clear_vision_memory():
    """
    Frees Python references and clears GPU cache for vision chat
    """
    global _VISION_CURRENT_MODEL, _VISION_CURRENT_TOKENIZER, _VISION_CHAT_HISTORY, _VISION_CURRENT_AGENT
    
    if _VISION_CURRENT_MODEL is not None:
        try:
            del _VISION_CURRENT_MODEL
            _VISION_CURRENT_MODEL = None
        except:
            pass
            
    if _VISION_CURRENT_TOKENIZER is not None:
        try:
            del _VISION_CURRENT_TOKENIZER
            _VISION_CURRENT_TOKENIZER = None
        except:
            pass

    # Clear vision processor
    clear_vision_processor()

    # Run garbage collection
    gc.collect()

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            print("Vision GPU memory cleared")
        except:
            pass

def load_vision_chat_model(model_name):
    """
    Load vision chat model with enhanced capabilities
    """
    # Initialize globals if they don't exist
    global _VISION_CURRENT_MODEL, _VISION_CURRENT_TOKENIZER, _VISION_CHAT_HISTORY, _VISION_CURRENT_AGENT
    
    if '_VISION_CURRENT_MODEL' not in globals():
        global _VISION_CURRENT_MODEL
        _VISION_CURRENT_MODEL = None
        
    if '_VISION_CURRENT_TOKENIZER' not in globals():
        global _VISION_CURRENT_TOKENIZER
        _VISION_CURRENT_TOKENIZER = None
        
    if '_VISION_CHAT_HISTORY' not in globals():
        global _VISION_CHAT_HISTORY
        _VISION_CHAT_HISTORY = []
        
    if '_VISION_CURRENT_AGENT' not in globals():
        global _VISION_CURRENT_AGENT
        _VISION_CURRENT_AGENT = None

    # Clear any previously loaded model
    clear_vision_memory()

    try:
        # Import necessary functions
        import torch
        from unsloth import FastLanguageModel

        # Load the model and tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(model_name)
        
        # Save references
        _VISION_CURRENT_MODEL = model
        _VISION_CURRENT_TOKENIZER = tokenizer

        return VisionChatModel(model, tokenizer)
    except Exception as e:
        print(f"Error loading vision model: {str(e)}")
        raise

class VisionChatModel:
    """
    Enhanced chat model with vision and knowledge base capabilities
    """
    def __init__(self, model, tokenizer, max_tokens=4096, temperature=0.4):
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Try to compile model if PyTorch >= 2.1
        try:
            self.model = torch.compile(self.model)
            print("Successfully compiled vision model with torch.compile()")
        except Exception as e:
            print(f"Could not compile vision model: {e}")
        
        # Use global history and agent tracking
        global _VISION_CHAT_HISTORY, _VISION_CURRENT_AGENT
        if _VISION_CURRENT_AGENT is None:
            _VISION_CURRENT_AGENT = None
        if _VISION_CHAT_HISTORY is None:
            _VISION_CHAT_HISTORY = []

    def generate_response_stream(self, user_input, agent=None, session_id=None, include_history=True):
        """Streams tokens from the model's response with knowledge base context and web search"""
        from transformers import TextIteratorStreamer
        import threading
        import time
        
        # Add user message to history
        self.add_to_history("user", user_input)
        
        # Check if agent changed
        agent_changed = self.check_agent_change(agent)
        
        # Get knowledge base context if session_id provided
        knowledge_context = ""
        relevant_sources = []
        web_search_results = []
        
        if session_id:
            try:
                kb = get_knowledge_base(session_id, agent)
                relevant_content = kb.get_relevant_content(user_input, max_results=3)
                
                if relevant_content:
                    knowledge_context = "\n\nRelevant Knowledge Base Content:\n"
                    knowledge_context += "=" * 50 + "\n"
                    
                    for i, item in enumerate(relevant_content, 1):
                        content_type = item.get('content_type', 'unknown')
                        best_chunk = item.get('best_chunk', '')
                        metadata = item.get('metadata', {})
                        score = item.get('best_score', 0)
                        
                        # Format source information
                        source_info = f"[Source {i}: {content_type.upper()}]"
                        if content_type == 'pdf_page':
                            page_num = metadata.get('page_number', 'unknown')
                            source_info += f" (Page {page_num})"
                        elif content_type == 'image':
                            img_name = metadata.get('image_name', 'unknown')
                            source_info += f" (Image: {img_name})"
                        
                        # Add relevance score
                        source_info += f" [Relevance: {score:.2f}]"
                        
                        knowledge_context += f"\n{source_info}\n"
                        knowledge_context += "-" * 30 + "\n"
                        knowledge_context += f"{best_chunk[:300]}...\n\n"
                        
                        # Store source information for the model
                        relevant_sources.append({
                            'type': content_type,
                            'content': best_chunk,
                            'metadata': metadata,
                            'score': score
                        })
                    
                    knowledge_context += "=" * 50 + "\n"
                
                # Perform web search if available
                if WEB_SEARCH_AVAILABLE and search_with_kb_context:
                    try:
                        print(f"[Vision SDK] Performing web search for query: {user_input}")
                        web_search_data = search_with_kb_context(
                            user_input, 
                            knowledge_context, 
                            max_results=3
                        )
                        
                        if web_search_data.get('search_performed') and web_search_data.get('web_results'):
                            web_search_results = web_search_data['web_results']
                            web_context = web_search_data['combined_context']
                            
                            # Update knowledge context with web search results
                            knowledge_context = web_context
                            
                            print(f"[Vision SDK] Web search completed. Found {len(web_search_results)} results.")
                        else:
                            print(f"[Vision SDK] Web search completed but no results found.")
                            
                    except Exception as web_e:
                        print(f"[Vision SDK] Web search error: {web_e}")
                        # Continue with knowledge base context only
                    
            except Exception as e:
                print(f"Error getting knowledge base context: {e}")
        
        # Build messages array with enhanced context
        messages = self._build_messages(
            user_input, 
            agent, 
            include_history and not agent_changed, 
            knowledge_context, 
            relevant_sources,
            web_search_results
        )
        
        # Convert to prompt
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenize prompt
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
            use_cache=True
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
        
    def _build_messages(self, user_input, agent, include_history, knowledge_context="", relevant_sources=None, web_search_results=None):
        """Build messages array for the model with knowledge context and web search results"""
        messages = []
        
        # Enhanced system message with embedding-based context and web search
        system_content = f"""\
You are a knowledgeable and helpful {agent} AI assistant with advanced visual and document understanding capabilities, enhanced with real-time web search integration. You should:
1. Provide clear, accurate, and concise answers based on the provided knowledge base content and web search results.
2. If you do not know the answer or cannot be certain, say so clearly.
3. Use simple language, but include necessary details.
4. When appropriate, provide step-by-step reasoning or explanations.
5. Maintain a polite and professional tone.
6. Avoid including irrelevant or sensitive information.
7. When referencing documents or images from the knowledge base, be specific about the source and page/image number.
8. If visual content is mentioned, describe what you understand from the images or charts.
9. Use the relevance scores to prioritize the most relevant information.
10. If multiple sources are provided, synthesize information from all relevant sources.
11. When using web search results, cite the sources appropriately and indicate if information is from recent web searches.
12. Combine knowledge base information with web search results to provide comprehensive answers.
13. Prioritize recent and authoritative web sources when available.

IMPORTANT: The context below includes both knowledge base content (retrieved using advanced embedding-based search) and real-time web search results. Each source includes relevance scores indicating how well it matches the user's query. Prioritize sources with higher relevance scores and recent web information.

{knowledge_context}

Now, please answer the following user query:
"""
        
        messages.append({
            "role": "system",
            "content": system_content
        })
        
        # Add conversation history if requested
        if include_history and len(_VISION_CHAT_HISTORY) > 0:
            for msg in _VISION_CHAT_HISTORY[-3:]:  # Last 3 messages
                messages.append(msg)
        
        # Add current user input
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        return messages
        
    def clear_history(self):
        """Clear the conversation history"""
        global _VISION_CHAT_HISTORY
        _VISION_CHAT_HISTORY = []

    def check_agent_change(self, agent):
        """Check if the agent has changed and clear history if needed"""
        global _VISION_CURRENT_AGENT, _VISION_CHAT_HISTORY
        
        if _VISION_CURRENT_AGENT is None:
            _VISION_CURRENT_AGENT = agent
            return False
            
        if agent != _VISION_CURRENT_AGENT:
            _VISION_CURRENT_AGENT = agent
            self.clear_history()
            return True
            
        return False
        
    def add_to_history(self, role, content):
        """Add a message to the conversation history"""
        global _VISION_CHAT_HISTORY
        _VISION_CHAT_HISTORY.append({"role": role, "content": content})

# Route handlers
@vision_chat_bp.route('/vision-chat-initialize', methods=['POST'])
def vision_chat_initialize():
    """Initialize vision chat session with knowledge base"""
    try:
        data = request.get_json()
        agent = data.get('agent')
        model_name = data.get('model_name', 'unsloth/Qwen3-1.7B-bnb-4bit')
        session_id = data.get('session_id')
        
        if not agent:
            return jsonify({'error': 'Agent type is required'}), 400
            
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Validate agent type
        valid_agents = ['general', 'tech', 'legal', 'finance', 'meetings', 'research', 'image-generator']
        if agent not in valid_agents:
            return jsonify({'error': f'Invalid agent type. Must be one of: {valid_agents}'}), 400
        
        # Initialize Vision LLM Processor for image processing
        print("🔄 Initializing Vision LLM Processor...")
        vision_processor_initialized = initialize_vision_processor()
        
        # Initialize knowledge base
        kb = get_knowledge_base(session_id, agent)
        
        # Clear vision memory
        clear_vision_memory()
        
        response = {
            'status': 'success',
            'message': f'{agent.title()} vision agent initialized successfully',
            'agent': agent,
            'model': model_name,
            'session_id': session_id,
            'vision_processor_initialized': vision_processor_initialized,
            'knowledge_base': kb.get_knowledge_base_summary()
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in vision chat initialize: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@vision_chat_bp.route('/vision-chat/add-document', methods=['POST'])
def add_document_to_kb():
    """Add document to knowledge base"""
    try:
        session_id = request.form.get('session_id')
        agent_id = request.form.get('agent_id')
        
        if not session_id or not agent_id:
            return jsonify({'error': 'Session ID and Agent ID are required'}), 400
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save file temporarily
        temp_path = os.path.join(tempfile.gettempdir(), secure_filename(file.filename))
        file.save(temp_path)
        
        try:
            # Get knowledge base
            kb = get_knowledge_base(session_id, agent_id)
            
            # Determine file type
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext == '.pdf':
                file_type = 'pdf'
            else:
                file_type = 'document'
            
            # Add document to knowledge base
            result = kb.add_document(temp_path, file_type)
            
            if 'error' in result:
                return jsonify({'error': result['error']}), 500
            
            return jsonify({
                'status': 'success',
                'message': 'Document added to knowledge base successfully',
                'document': result
            })
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
    except Exception as e:
        print(f"Error adding document: {str(e)}")
        return jsonify({'error': str(e)}), 500

@vision_chat_bp.route('/vision-chat/add-image', methods=['POST'])
def add_image_to_kb():
    """Add image to knowledge base"""
    try:
        session_id = request.form.get('session_id')
        agent_id = request.form.get('agent_id')
        
        if not session_id or not agent_id:
            return jsonify({'error': 'Session ID and Agent ID are required'}), 400
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if it's an image file
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
            return jsonify({'error': 'Invalid image file type'}), 400
        
        # Save file temporarily
        temp_path = os.path.join(tempfile.gettempdir(), secure_filename(file.filename))
        file.save(temp_path)
        
        try:
            # Get knowledge base
            kb = get_knowledge_base(session_id, agent_id)
            
            # Add image to knowledge base
            result = kb.add_image(temp_path)
            
            if 'error' in result:
                return jsonify({'error': result['error']}), 500
            
            return jsonify({
                'status': 'success',
                'message': 'Image added to knowledge base successfully',
                'image': result
            })
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
    except Exception as e:
        print(f"Error adding image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@vision_chat_bp.route('/vision-chat/knowledge-base/<session_id>', methods=['GET'])
def get_knowledge_base_info(session_id):
    """Get knowledge base information"""
    try:
        agent_id = request.args.get('agent_id')
        if not agent_id:
            return jsonify({'error': 'Agent ID is required'}), 400
        
        kb = get_knowledge_base(session_id, agent_id)
        return jsonify(kb.get_knowledge_base_summary())
        
    except Exception as e:
        print(f"Error getting knowledge base info: {str(e)}")
        return jsonify({'error': str(e)}), 500

@vision_chat_bp.route('/vision-chat', methods=['POST'])
def vision_chat():
    """Main vision chat endpoint with knowledge base integration"""
    global vision_rag_chat, _VISION_CHAT_HISTORY, _VISION_CURRENT_AGENT
    
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400
    
    user_input = data['message']
    agent = data['agent']
    model = data['model']
    session_id = data.get('session_id')
    system_command = data.get('system_command')
    
    # Handle system commands
    if system_command == 'initialize' or user_input.startswith('SYSTEM_INIT:'):
        print(f"[Vision SDK] Initializing agent with cleanup: {agent}")
        
        try:
            # Initialize Vision LLM Processor for image processing
            print("🔄 Initializing Vision LLM Processor...")
            vision_processor_initialized = initialize_vision_processor()
            
            # Clear vision memory
            clear_vision_memory()
            
            # Initialize knowledge base if session_id provided
            if session_id:
                kb = get_knowledge_base(session_id, agent)
                print(f"[Vision SDK] Knowledge base initialized for session: {session_id}")
            
            return jsonify({
                'status': 'success',
                'message': f'{agent.title()} vision agent initialized successfully with cleanup',
                'agent': agent,
                'model': model,
                'session_id': session_id,
                'vision_processor_initialized': vision_processor_initialized
            })
            
        except Exception as init_error:
            print(f"[Vision SDK] Initialization error: {str(init_error)}")
            return jsonify({
                'status': 'error',
                'message': f'Initialization failed: {str(init_error)}',
                'agent': agent
            }), 500
    
    # Check if we should clear history
    if data.get('clear_history', False):
        if '_VISION_CHAT_HISTORY' not in globals():
            global _VISION_CHAT_HISTORY
            _VISION_CHAT_HISTORY = []
        else:
            _VISION_CHAT_HISTORY = []
        return jsonify({"message": "Vision chat history cleared"}), 200
    
    # Check if we should include history
    include_history = data.get('include_history', True)
    
    def generate():
        try:
            global vision_rag_chat
            
            # Start event
            print("\nVision User:", user_input)
            print("\nGenerating vision response...")
            yield 'data: {"status": "start"}\n\n'

            if vision_rag_chat is None:
                # Create a fallback model when vision_rag_chat is None
                try:
                    fallback_model = load_vision_chat_model(model)
                    print("Vision request received")
                    
                    # Check for agent change and add history management
                    fallback_model.check_agent_change(agent)
                    
                    full_response = ""
                    for chunk, tokens_per_second in fallback_model.generate_response_stream(
                        user_input, 
                        agent, 
                        session_id,
                        include_history=include_history
                    ):
                        full_response += chunk
                        print(chunk, end='', flush=True)
                        response_data = {
                            "response": chunk,
                            "tokens_per_second": round(tokens_per_second, 2)
                        }
                        yield f'data: {json.dumps(response_data)}\n\n'
                except Exception as e:
                    print(f"\nError in vision fallback model: {str(e)}")
                    error_data = {"error": str(e), "status": "error"}
                    yield f'data: {json.dumps(error_data)}\n\n'
                    return
            else:
                # Use the existing vision_rag_chat model
                full_response = ""
                for chunk, tokens_per_second in vision_rag_chat.generate_response_stream(
                    user_input,
                    agent,
                    include_history=include_history
                ):
                    full_response += chunk
                    print(chunk, end='', flush=True)
                    response_data = {
                        "response": chunk,
                        "tokens_per_second": round(tokens_per_second, 2)
                    }
                    yield f'data: {json.dumps(response_data)}\n\n'
            
            print("\n\nVision response complete.\n")
            yield f'data: {json.dumps({"response": full_response, "replace": True})}\n\n'
            yield 'data: {"status": "complete"}\n\n'
            
        except Exception as e:
            error_msg = f"\nError generating vision response: {str(e)}"
            print(error_msg)
            error_data = {"error": str(e), "status": "error"}
            yield f'data: {json.dumps(error_data)}\n\n'
         
    return Response(generate(), mimetype='text/event-stream')

@vision_chat_bp.route('/vision-chat/history', methods=['GET'])
def get_vision_chat_history():
    """Get vision chat history"""
    global vision_rag_chat
    if vision_rag_chat is None:
        return jsonify({"error": "Vision chat model not initialized"}), 400
    
    return jsonify({
        "history": vision_rag_chat.chat_history.history if hasattr(vision_rag_chat, 'chat_history') else []
    })

@vision_chat_bp.route('/vision-chat/test', methods=['POST'])
def test_vision_chat():
    """Test endpoint for vision chat debugging"""
    print("=== ENTERING test_vision_chat ===")
    print(f"Request method: {request.method}")
    print(f"Request URL: {request.url}")
    
    try:
        print("Getting JSON data...")
        data = request.get_json()
        print(f"Vision test endpoint received: {data}")
        
        response = {
            'status': 'success',
            'message': 'Vision chat test endpoint working',
            'received_data': data,
            'timestamp': datetime.now().isoformat()
        }
        print(f"Returning response: {response}")
        return jsonify(response)
        
    except Exception as e:
        print(f"=== EXCEPTION in test_vision_chat ===")
        print(f"Exception type: {type(e)}")
        print(f"Exception message: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=== END EXCEPTION ===")
        return jsonify({'error': str(e)}), 500

@vision_chat_bp.route('/vision-chat/rebuild-index/<session_id>', methods=['POST'])
def rebuild_embedding_index(session_id):
    """Rebuild the embedding index for a knowledge base"""
    try:
        agent_id = request.json.get('agent_id')
        if not agent_id:
            return jsonify({'error': 'Agent ID is required'}), 400
        
        kb = get_knowledge_base(session_id, agent_id)
        
        # Check if embedding system is available
        if kb.embedding_model is None:
            return jsonify({'error': 'Embedding system not available'}), 400
        
        # Clear existing index
        kb.faiss_index = faiss.IndexFlatL2(kb.embedding_dimension)
        kb.chunk_mapping = {}
        kb.embedding_metadata = {}
        
        # Rebuild index from all processed content
        total_chunks = 0
        
        for content_id, content_info in kb.metadata.get('processed_content', {}).items():
            try:
                processed_file = content_info.get('file')
                if os.path.exists(processed_file):
                    with open(processed_file, 'r') as f:
                        content_data = json.load(f)
                    
                    # Re-add content to index based on type
                    if content_info['type'] == 'document':
                        if content_data.get('type') == 'pdf':
                            # Re-add PDF pages
                            for page_num, page_data in content_data.get('pages', {}).items():
                                page_text = page_data.get('text', '') + ' ' + page_data.get('enhanced_content', '')
                                kb._add_content_to_index(
                                    content_id=content_id,
                                    content_type='pdf_page',
                                    text_content=page_text,
                                    metadata={
                                        'page_number': page_num,
                                        'document_name': content_data.get('doc_id', ''),
                                        'has_images': len(page_data.get('images', [])) > 0
                                    }
                                )
                                total_chunks += 1
                        else:
                            # Re-add general document
                            content = content_data.get('content', '')
                            kb._add_content_to_index(
                                content_id=content_id,
                                content_type='document',
                                text_content=content,
                                metadata={
                                    'document_name': content_data.get('doc_id', ''),
                                    'document_type': content_data.get('type', '')
                                }
                            )
                            total_chunks += 1
                    
                    elif content_info['type'] == 'image':
                        # Re-add image description
                        description = content_data.get('description', '')
                        if description:
                            kb._add_content_to_index(
                                content_id=content_id,
                                content_type='image',
                                text_content=description,
                                metadata={
                                    'image_name': content_data.get('img_id', ''),
                                    'image_path': content_data.get('path', ''),
                                    'content_type': 'image_description'
                                }
                            )
                            total_chunks += 1
                            
            except Exception as e:
                print(f"Error rebuilding index for content {content_id}: {e}")
                continue
        
        # Save the rebuilt index
        kb._save_index()
        
        return jsonify({
            'status': 'success',
            'message': f'Embedding index rebuilt successfully',
            'total_chunks': total_chunks,
            'index_size': kb.faiss_index.ntotal if kb.faiss_index else 0
        })
        
    except Exception as e:
        print(f"Error rebuilding index: {str(e)}")
        return jsonify({'error': str(e)}), 500

@vision_chat_bp.route('/vision-chat/search/<session_id>', methods=['POST'])
def search_knowledge_base(session_id):
    """Search the knowledge base directly"""
    try:
        data = request.get_json()
        query = data.get('query')
        agent_id = data.get('agent_id')
        max_results = data.get('max_results', 5)
        
        if not query or not agent_id:
            return jsonify({'error': 'Query and Agent ID are required'}), 400
        
        kb = get_knowledge_base(session_id, agent_id)
        results = kb.get_relevant_content(query, max_results=max_results)
        
        # Format results for response
        formatted_results = []
        for result in results:
            formatted_result = {
                'content_id': result.get('content_id'),
                'content_type': result.get('content_type'),
                'best_chunk': result.get('best_chunk', ''),
                'relevance_score': result.get('best_score', 0),
                'metadata': result.get('metadata', {}),
                'all_chunks_count': len(result.get('all_chunks', [])),
                'total_score': result.get('total_score', 0)
            }
            formatted_results.append(formatted_result)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': formatted_results,
            'total_results': len(formatted_results)
        })
        
    except Exception as e:
        print(f"Error searching knowledge base: {str(e)}")
        return jsonify({'error': str(e)}), 500

@vision_chat_bp.route('/vision-chat/index-stats/<session_id>', methods=['GET'])
def get_index_statistics(session_id):
    """Get statistics about the embedding index"""
    try:
        agent_id = request.args.get('agent_id')
        if not agent_id:
            return jsonify({'error': 'Agent ID is required'}), 400
        
        kb = get_knowledge_base(session_id, agent_id)
        
        # Get basic stats
        stats = {
            'session_id': session_id,
            'agent_id': agent_id,
            'embedding_model_available': kb.embedding_model is not None,
            'faiss_index_available': kb.faiss_index is not None,
            'embedding_dimension': kb.embedding_dimension,
            'total_chunks': kb.faiss_index.ntotal if kb.faiss_index else 0,
            'total_documents': len(kb.metadata.get('documents', {})),
            'total_images': len(kb.metadata.get('images', {})),
            'total_processed': len(kb.metadata.get('processed_content', {})),
            'index_file_exists': os.path.exists(os.path.join(kb.index_path, "faiss_index.faiss")),
            'mapping_file_exists': os.path.exists(os.path.join(kb.index_path, "chunk_mapping.json")),
            'metadata_file_exists': os.path.exists(os.path.join(kb.index_path, "embedding_metadata.json"))
        }
        
        # Get content type distribution
        if kb.embedding_metadata:
            content_types = {}
            for chunk_meta in kb.embedding_metadata.values():
                content_type = chunk_meta.get('content_type', 'unknown')
                content_types[content_type] = content_types.get(content_type, 0) + 1
            
            stats['content_type_distribution'] = content_types
        
        return jsonify(stats)
        
    except Exception as e:
        print(f"Error getting index statistics: {str(e)}")
        return jsonify({'error': str(e)}), 500

@vision_chat_bp.route('/vision-chat/clear-index/<session_id>', methods=['POST'])
def clear_embedding_index(session_id):
    """Clear the embedding index for a knowledge base"""
    try:
        agent_id = request.json.get('agent_id')
        if not agent_id:
            return jsonify({'error': 'Agent ID is required'}), 400
        
        kb = get_knowledge_base(session_id, agent_id)
        
        # Clear index
        if kb.faiss_index is not None:
            kb.faiss_index = faiss.IndexFlatL2(kb.embedding_dimension)
        kb.chunk_mapping = {}
        kb.embedding_metadata = {}
        
        # Save empty index
        kb._save_index()
        
        return jsonify({
            'status': 'success',
            'message': 'Embedding index cleared successfully'
        })
        
    except Exception as e:
        print(f"Error clearing index: {str(e)}")
        return jsonify({'error': str(e)}), 500

@vision_chat_bp.route('/vision-chat/delete-document/<session_id>', methods=['DELETE'])
def delete_document_from_kb(session_id):
    """Delete document from knowledge base"""
    try:
        data = request.get_json()
        doc_id = data.get('doc_id')
        agent_id = data.get('agent_id')
        
        if not doc_id or not agent_id:
            return jsonify({'error': 'Document ID and Agent ID are required'}), 400
        
        kb = get_knowledge_base(session_id, agent_id)
        result = kb.delete_document(doc_id)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 400
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error deleting document: {str(e)}")
        return jsonify({'error': str(e)}), 500

@vision_chat_bp.route('/vision-chat/delete-image/<session_id>', methods=['DELETE'])
def delete_image_from_kb(session_id):
    """Delete image from knowledge base"""
    try:
        data = request.get_json()
        img_id = data.get('img_id')
        agent_id = data.get('agent_id')
        
        if not img_id or not agent_id:
            return jsonify({'error': 'Image ID and Agent ID are required'}), 400
        
        kb = get_knowledge_base(session_id, agent_id)
        result = kb.delete_image(img_id)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 400
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error deleting image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@vision_chat_bp.route('/vision-chat/delete-page/<session_id>', methods=['DELETE'])
def delete_page_from_document(session_id):
    """Delete specific page from document"""
    try:
        data = request.get_json()
        doc_id = data.get('doc_id')
        page_number = data.get('page_number')
        agent_id = data.get('agent_id')
        
        if not doc_id or not page_number or not agent_id:
            return jsonify({'error': 'Document ID, Page Number, and Agent ID are required'}), 400
        
        kb = get_knowledge_base(session_id, agent_id)
        result = kb.delete_page_from_document(doc_id, page_number)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 400
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error deleting page: {str(e)}")
        return jsonify({'error': str(e)}), 500

@vision_chat_bp.route('/vision-chat/clear-kb/<session_id>', methods=['DELETE'])
def clear_knowledge_base_endpoint(session_id):
    """Clear entire knowledge base"""
    try:
        agent_id = request.json.get('agent_id')
        if not agent_id:
            return jsonify({'error': 'Agent ID is required'}), 400
        
        kb = get_knowledge_base(session_id, agent_id)
        result = kb.clear_knowledge_base()
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 400
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error clearing knowledge base: {str(e)}")
        return jsonify({'error': str(e)}), 500

@vision_chat_bp.route('/vision-chat/bulk-delete/<session_id>', methods=['DELETE'])
def bulk_delete_content_endpoint(session_id):
    """Bulk delete multiple content items"""
    try:
        data = request.get_json()
        content_ids = data.get('content_ids', [])
        content_type = data.get('content_type', 'document')
        agent_id = data.get('agent_id')
        
        if not content_ids or not agent_id:
            return jsonify({'error': 'Content IDs and Agent ID are required'}), 400
        
        if not isinstance(content_ids, list):
            return jsonify({'error': 'Content IDs must be a list'}), 400
        
        kb = get_knowledge_base(session_id, agent_id)
        result = kb.bulk_delete_content(content_ids, content_type)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 400
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in bulk delete: {str(e)}")
        return jsonify({'error': str(e)}), 500

@vision_chat_bp.route('/vision-chat/deletion-preview/<session_id>', methods=['POST'])
def get_deletion_preview_endpoint(session_id):
    """Get preview of what will be deleted"""
    try:
        data = request.get_json()
        content_id = data.get('content_id')
        content_type = data.get('content_type', 'document')
        agent_id = data.get('agent_id')
        
        if not content_id or not agent_id:
            return jsonify({'error': 'Content ID and Agent ID are required'}), 400
        
        kb = get_knowledge_base(session_id, agent_id)
        preview = kb.get_deletion_preview(content_id, content_type)
        
        if 'error' in preview:
            return jsonify({'error': preview['error']}), 400
        
        return jsonify(preview)
        
    except Exception as e:
        print(f"Error getting deletion preview: {str(e)}")
        return jsonify({'error': str(e)}), 500

@vision_chat_bp.route('/vision-chat/list-content/<session_id>', methods=['GET'])
def list_knowledge_base_content(session_id):
    """List all content in knowledge base"""
    try:
        agent_id = request.args.get('agent_id')
        if not agent_id:
            return jsonify({'error': 'Agent ID is required'}), 400
        
        kb = get_knowledge_base(session_id, agent_id)
        
        # Get documents
        documents = []
        for doc_id, doc_info in kb.metadata.get('documents', {}).items():
            documents.append({
                'id': doc_id,
                'name': doc_info.get('name', ''),
                'type': doc_info.get('type', ''),
                'added_at': doc_info.get('added_at', ''),
                'pages': doc_info.get('pages', 0),
                'path': doc_info.get('path', '')
            })
        
        # Get images
        images = []
        for img_id, img_info in kb.metadata.get('images', {}).items():
            images.append({
                'id': img_id,
                'name': img_info.get('name', ''),
                'added_at': img_info.get('added_at', ''),
                'description': img_info.get('description', ''),
                'path': img_info.get('path', '')
            })
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'agent_id': agent_id,
            'documents': documents,
            'images': images,
            'total_documents': len(documents),
            'total_images': len(images),
            'total_chunks': kb.faiss_index.ntotal if kb.faiss_index else 0
        })
        
    except Exception as e:
        print(f"Error listing content: {str(e)}")
        return jsonify({'error': str(e)}), 500

@vision_chat_bp.route('/vision-chat/content-details/<session_id>', methods=['GET'])
def get_content_details(session_id):
    """Get detailed information about specific content"""
    try:
        content_id = request.args.get('content_id')
        content_type = request.args.get('content_type', 'document')
        agent_id = request.args.get('agent_id')
        
        if not content_id or not agent_id:
            return jsonify({'error': 'Content ID and Agent ID are required'}), 400
        
        kb = get_knowledge_base(session_id, agent_id)
        
        if content_type == 'document':
            if content_id not in kb.metadata.get('documents', {}):
                return jsonify({'error': 'Document not found'}), 404
            
            doc_info = kb.metadata['documents'][content_id]
            processed_file = kb.metadata.get('processed_content', {}).get(content_id, {}).get('file', '')
            
            details = {
                'id': content_id,
                'type': 'document',
                'name': doc_info.get('name', ''),
                'file_type': doc_info.get('type', ''),
                'added_at': doc_info.get('added_at', ''),
                'pages': doc_info.get('pages', 0),
                'path': doc_info.get('path', ''),
                'processed': doc_info.get('processed', False),
                'content_summary': doc_info.get('content_summary', '')
            }
            
            # Add processed content if available
            if processed_file and os.path.exists(processed_file):
                try:
                    with open(processed_file, 'r') as f:
                        content_data = json.load(f)
                    details['processed_content'] = content_data
                except Exception as e:
                    details['processed_content_error'] = str(e)
            
        elif content_type == 'image':
            if content_id not in kb.metadata.get('images', {}):
                return jsonify({'error': 'Image not found'}), 404
            
            img_info = kb.metadata['images'][content_id]
            processed_file = kb.metadata.get('processed_content', {}).get(content_id, {}).get('file', '')
            
            details = {
                'id': content_id,
                'type': 'image',
                'name': img_info.get('name', ''),
                'added_at': img_info.get('added_at', ''),
                'description': img_info.get('description', ''),
                'path': img_info.get('path', ''),
                'processed': img_info.get('processed', False)
            }
            
            # Add processed content if available
            if processed_file and os.path.exists(processed_file):
                try:
                    with open(processed_file, 'r') as f:
                        content_data = json.load(f)
                    details['processed_content'] = content_data
                except Exception as e:
                    details['processed_content_error'] = str(e)
        
        else:
            return jsonify({'error': 'Invalid content type'}), 400
        
        # Add chunk information
        chunks = []
        for chunk_id, chunk_meta in kb.embedding_metadata.items():
            if chunk_meta.get('content_id') == content_id:
                chunks.append({
                    'chunk_id': chunk_id,
                    'content_type': chunk_meta.get('content_type', ''),
                    'chunk_index': chunk_meta.get('chunk_index', 0),
                    'metadata': chunk_meta.get('metadata', {})
                })
        
        details['chunks'] = chunks
        details['total_chunks'] = len(chunks)
        
        return jsonify(details)
        
    except Exception as e:
        print(f"Error getting content details: {str(e)}")
        return jsonify({'error': str(e)}), 500

@vision_chat_bp.route('/vision-chat/health-check/<session_id>', methods=['GET'])
def knowledge_base_health_check(session_id):
    """Health check for knowledge base and vision processor"""
    try:
        agent_id = request.args.get('agent_id')
        if not agent_id:
            return jsonify({'error': 'Agent ID is required'}), 400
        
        kb = get_knowledge_base(session_id, agent_id)
        
        health_status = {
            'session_id': session_id,
            'agent_id': agent_id,
            'status': 'healthy',
            'issues': [],
            'warnings': [],
            'checks': {}
        }
        
        # Check embedding system
        health_status['checks']['embedding_system'] = {
            'model_available': kb.embedding_model is not None,
            'index_available': kb.faiss_index is not None,
            'index_size': kb.faiss_index.ntotal if kb.faiss_index else 0
        }
        
        if kb.embedding_model is None:
            health_status['warnings'].append('Embedding model not available')
        
        if kb.faiss_index is None:
            health_status['warnings'].append('FAISS index not available')
        
        # Check vision processor status
        vision_processor = get_vision_processor()
        vision_status = {
            'initialized': vision_processor is not None,
            'model_loaded': vision_processor is not None and vision_processor.model is not None,
            'device': None,
            'model_type': None
        }
        
        if vision_processor is not None:
            vision_status['device'] = getattr(vision_processor, 'device', 'unknown')
            vision_status['model_type'] = 'SmolVLM-500M-Instruct'
            
            # Check GPU memory if using CUDA
            if vision_status['device'] == 'cuda':
                try:
                    import torch
                    vision_status['gpu_memory'] = {
                        'allocated_gb': torch.cuda.memory_allocated(0) / 1024**3,
                        'reserved_gb': torch.cuda.memory_reserved(0) / 1024**3,
                        'total_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3
                    }
                except Exception as e:
                    vision_status['gpu_memory_error'] = str(e)
        
        health_status['checks']['vision_processor'] = vision_status

        # Check file integrity
        missing_files = []
        orphaned_files = []
        
        # Check document files
        for doc_id, doc_info in kb.metadata.get('documents', {}).items():
            doc_path = doc_info.get('path', '')
            if not os.path.exists(doc_path):
                missing_files.append(f"Document file: {doc_path}")
                health_status['status'] = 'degraded'
        
        # Check image files
        for img_id, img_info in kb.metadata.get('images', {}).items():
            img_path = img_info.get('path', '')
            if not os.path.exists(img_path):
                missing_files.append(f"Image file: {img_path}")
                health_status['status'] = 'degraded'
        
        # Check processed content files
        for content_id, content_info in kb.metadata.get('processed_content', {}).items():
            processed_file = content_info.get('file', '')
            if not os.path.exists(processed_file):
                missing_files.append(f"Processed file: {processed_file}")
                health_status['status'] = 'degraded'
        
        # Check for orphaned files
        for filename in os.listdir(kb.documents_path):
            file_path = os.path.join(kb.documents_path, filename)
            if os.path.isfile(file_path):
                # Check if file is referenced in metadata
                referenced = False
                for doc_info in kb.metadata.get('documents', {}).values():
                    if doc_info.get('path') == file_path:
                        referenced = True
                        break
                
                if not referenced:
                    orphaned_files.append(f"Orphaned document: {filename}")
        
        for filename in os.listdir(kb.images_path):
            file_path = os.path.join(kb.images_path, filename)
            if os.path.isfile(file_path):
                # Check if file is referenced in metadata
                referenced = False
                for img_info in kb.metadata.get('images', {}).values():
                    if img_info.get('path') == file_path:
                        referenced = True
                        break
                
                if not referenced:
                    orphaned_files.append(f"Orphaned image: {filename}")
        
        health_status['checks']['file_integrity'] = {
            'missing_files': missing_files,
            'orphaned_files': orphaned_files,
            'total_missing': len(missing_files),
            'total_orphaned': len(orphaned_files)
        }
        
        if missing_files:
            health_status['issues'].extend(missing_files)
        
        if orphaned_files:
            health_status['warnings'].extend(orphaned_files)
        
        # Check index consistency
        index_chunks = kb.faiss_index.ntotal if kb.faiss_index else 0
        mapping_chunks = len(kb.chunk_mapping)
        metadata_chunks = len(kb.embedding_metadata)
        
        health_status['checks']['index_consistency'] = {
            'faiss_chunks': index_chunks,
            'mapping_chunks': mapping_chunks,
            'metadata_chunks': metadata_chunks,
            'consistent': index_chunks == mapping_chunks == metadata_chunks
        }
        
        if not health_status['checks']['index_consistency']['consistent']:
            health_status['issues'].append('Index inconsistency detected')
            health_status['status'] = 'degraded'
        
        # Overall status
        if health_status['issues']:
            health_status['status'] = 'unhealthy'
        elif health_status['warnings']:
            health_status['status'] = 'degraded'
        
        return jsonify(health_status)
        
    except Exception as e:
        print(f"Error in health check: {str(e)}")
        return jsonify({'error': str(e)}), 500

@vision_chat_bp.route('/vision-chat/web-search-status', methods=['GET'])
def get_web_search_status():
    """Get web search functionality status"""
    try:
        status = {
            'web_search_available': WEB_SEARCH_AVAILABLE,
            'perplexity_api_configured': bool(os.getenv("PERPLEXITY_API_KEY", "")),
            'web_search_enabled': True,  # Can be made configurable
            'timestamp': datetime.now().isoformat()
        }
        
        if WEB_SEARCH_AVAILABLE:
            try:
                manager = get_web_search_manager()
                status['cache_size'] = len(manager.cache)
                status['cache_ttl'] = manager.cache_ttl
            except Exception as e:
                status['manager_error'] = str(e)
        
        return jsonify(status)
        
    except Exception as e:
        print(f"Error getting web search status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@vision_chat_bp.route('/vision-chat/web-search/test', methods=['POST'])
def test_web_search():
    """Test web search functionality"""
    try:
        data = request.get_json()
        query = data.get('query', 'test query')
        max_results = data.get('max_results', 3)
        
        if not WEB_SEARCH_AVAILABLE:
            return jsonify({
                'error': 'Web search not available',
                'web_search_available': False
            }), 400
        
        # Test web search
        results = search_web_for_query(query, max_results=max_results)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results_count': len(results),
            'results': [
                {
                    'title': result.title,
                    'content': result.content[:200] + "..." if len(result.content) > 200 else result.content,
                    'url': result.url,
                    'relevance_score': result.relevance_score,
                    'source': result.source
                }
                for result in results
            ],
            'web_search_available': True
        })
        
    except Exception as e:
        print(f"Error testing web search: {str(e)}")
        return jsonify({'error': str(e)}), 500

@vision_chat_bp.route('/vision-chat/web-search/clear-cache', methods=['POST'])
def clear_web_search_cache():
    """Clear web search cache"""
    try:
        if not WEB_SEARCH_AVAILABLE:
            return jsonify({
                'error': 'Web search not available',
                'web_search_available': False
            }), 400
        
        manager = get_web_search_manager()
        manager.clear_cache()
        
        return jsonify({
            'status': 'success',
            'message': 'Web search cache cleared successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error clearing web search cache: {str(e)}")
        return jsonify({'error': str(e)}), 500

@vision_chat_bp.route('/vision-chat/vision-processor-status', methods=['GET'])
def get_vision_processor_status():
    """Get detailed status of the vision processor"""
    try:
        vision_processor = get_vision_processor()
        
        status = {
            'initialized': vision_processor is not None,
            'model_loaded': vision_processor is not None and vision_processor.model is not None,
            'processor_loaded': vision_processor is not None and vision_processor.processor is not None,
            'device': None,
            'model_type': None,
            'timestamp': datetime.now().isoformat()
        }
        
        if vision_processor is not None:
            status['device'] = getattr(vision_processor, 'device', 'unknown')
            status['model_type'] = 'SmolVLM-500M-Instruct'
            
            # Check GPU memory if using CUDA
            if status['device'] == 'cuda':
                try:
                    import torch
                    if torch.cuda.is_available():
                        status['gpu_info'] = {
                            'device_name': torch.cuda.get_device_name(0),
                            'cuda_version': torch.version.cuda,
                            'memory_allocated_gb': torch.cuda.memory_allocated(0) / 1024**3,
                            'memory_reserved_gb': torch.cuda.memory_reserved(0) / 1024**3,
                            'memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3
                        }
                except Exception as e:
                    status['gpu_info_error'] = str(e)
        
        return jsonify(status)
        
    except Exception as e:
        print(f"Error getting vision processor status: {str(e)}")
        return jsonify({'error': str(e)}), 500
