# sdk\decompute\constants.py
"""Constants used throughout the Decompute SDK with file upload support and streaming."""

# API Endpoints
CHAT_INITIALIZE = '/chat-initialize-sdk'
LOAD_EXISTING_MODEL = '/load-existing-model'
CHAT = '/chat'
CHAT_STREAM = '/chat'  # Same endpoint, but using SSE for streaming
INITIALIZE_RAG = '/initialize-rag'
FINE_TUNING_PROGRESS = '/fine-tuning-progress'
SAVE_MODEL_STATE = '/api/save-model-state'
CHECK_FILENAME_EXISTS = '/api/check-filename-exists'
UPDATE_QUERY = '/api/update-query'
GENERATE_IMAGE = '/generate'
GET_IMAGE = '/images/'
SAVE_CONVERSATION = '/api/save-conversation/'
SAVE_FEEDBACK = '/api/save-feedback'
GET_SUGGESTIONS = '/api/get-suggestions'

# Streaming Endpoints
TRAINING_PROGRESS_STREAM = '/fine-tuning-progress'
IMAGE_GENERATION_STREAM = '/generate'
MODEL_LOADING_STREAM = '/model-loading-stream'

# Agent Types
AGENT_GENERAL = 'general'
AGENT_TECH = 'tech'
AGENT_LEGAL = 'legal'
AGENT_FINANCE = 'finance'
AGENT_MEETINGS = 'meetings'
AGENT_RESEARCH = 'research'
AGENT_IMAGE = 'image-generator'

AGENT_TYPES = [
    AGENT_GENERAL,
    AGENT_TECH,
    AGENT_LEGAL,
    AGENT_FINANCE,
    AGENT_MEETINGS,
    AGENT_RESEARCH,
    AGENT_IMAGE
]

# Storage Types
STORAGE_MODEL_MEMORY = 'model_memory'
STORAGE_SAVED_FILES = 'saved_files'

# File Upload Configuration
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB in bytes

# Allowed file types by agent (enhanced with feature flags support)
ALLOWED_FILE_TYPES = {
    AGENT_GENERAL: ['.pdf', '.docx', '.txt', '.md'],
    AGENT_TECH: ['.pdf', '.docx', '.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.yaml', '.yml'],
    AGENT_LEGAL: ['.pdf', '.docx', '.txt', '.md'],
    AGENT_FINANCE: ['.pdf', '.docx', '.txt', '.md', '.xlsx', '.xls', '.csv'],
    AGENT_MEETINGS: ['.pdf', '.docx', '.txt', '.md', '.wav', '.mp3', '.m4a', '.flac'],
    AGENT_RESEARCH: ['.pdf', '.docx', '.txt', '.md'],
    AGENT_IMAGE: ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.pdf', '.txt']
}

# File size limits by type
FILE_SIZE_LIMITS = {
    'documents': 50 * 1024 * 1024,  # 50MB for documents
    'images': 20 * 1024 * 1024,     # 20MB for images
    'audio': 100 * 1024 * 1024,     # 100MB for audio
    'code': 10 * 1024 * 1024,       # 10MB for code files
    'spreadsheets': 30 * 1024 * 1024 # 30MB for spreadsheets
}

# MIME type mappings
SUPPORTED_MIME_TYPES = {
    '.pdf': 'application/pdf',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.txt': 'text/plain',
    '.md': 'text/markdown',
    '.py': 'text/x-python',
    '.js': 'application/javascript',
    '.html': 'text/html',
    '.css': 'text/css',
    '.json': 'application/json',
    '.xml': 'application/xml',
    '.yaml': 'application/x-yaml',
    '.yml': 'application/x-yaml',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.xls': 'application/vnd.ms-excel',
    '.csv': 'text/csv',
    '.wav': 'audio/wav',
    '.mp3': 'audio/mpeg',
    '.m4a': 'audio/mp4',
    '.flac': 'audio/flac',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.gif': 'image/gif',
    '.bmp': 'image/bmp',
    '.tiff': 'image/tiff'
}

# SSE Event types
SSE_EVENT_TYPES = [
    'message',
    'error',
    'open',
    'close'
]

# SSE Status codes
SSE_STATUS_CODES = {
    'start': 'started',
    'error': 'error',
    'complete': 'complete',
    'progress': 'progress'
}