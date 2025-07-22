"""
File upload and document processing service for the Decompute SDK.
"""

import os
import mimetypes
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from oss_utils.http_client import HTTPClient
from oss_utils.constants import INITIALIZE_RAG, MAX_FILE_SIZE
from oss_utils.errors import FileProcessingError
from oss_utils.feature_flags import require_feature
import platform

# Backend-aligned allowed extensions
ALLOWED_EXTENSIONS = {
    '.txt', '.json', '.pdf', '.docx', '.wav', '.xlsx', '.xls', '.m4a', '.mp3', '.py', '.js'
}

DEFAULT_AGENT_TYPE = "general"
if platform.system().lower() == "windows":
    DEFAULT_MODEL = "unsloth/Qwen3-1.7B-bnb-4bit"
else:
    DEFAULT_MODEL = "mlx-community/Qwen3-4B-4bit"

class FileService:
    """Handles file upload and document processing operations."""
    def __init__(self, http_client: HTTPClient, logger=None):
        self.http_client = http_client
        self.logger = logger

    def _get_mime_type(self, file_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or 'application/octet-stream'

    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        path = Path(file_path)
        if not path.exists():
            return False, f"File does not exist: {file_path}"
        if not path.is_file():
            return False, f"Path is not a file: {file_path}"
        file_size = path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            size_mb = file_size / (1024 * 1024)
            max_mb = MAX_FILE_SIZE / (1024 * 1024)
            return False, f"File too large: {size_mb:.1f}MB (max: {max_mb:.1f}MB)"
        file_ext = path.suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            return False, f"Unsupported file type '{file_ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        try:
            with open(file_path, 'rb') as f:
                f.read(1)
        except Exception as e:
            return False, f"Error reading file: {str(e)}"
        return True, "File validation passed"

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileProcessingError(f"File does not exist: {file_path}", file_path=file_path)
        stat = path.stat()
        return {
            'path': str(path.absolute()),
            'name': path.name,
            'extension': path.suffix.lower(),
            'size_bytes': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'mime_type': self._get_mime_type(file_path),
            'is_supported': path.suffix.lower() in ALLOWED_EXTENSIONS,
            'created': stat.st_ctime,
            'modified': stat.st_mtime
        }

    @require_feature("file_upload")
    def upload_single_file(
        self,
        file_path: str,
        agent_type: str = DEFAULT_AGENT_TYPE,
        model: str = DEFAULT_MODEL,
        storage_type: str = 'saved_files',
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Upload a single file for RAG processing.
        Defaults:
            agent_type: 'general'
            model: platform-specific (Windows: 'unsloth/Qwen3-1.7B-bnb-4bit', Mac: 'mlx-community/Qwen3-4B-4bit')
        """
        if options is None:
            options = {}
        is_valid, validation_message = self.validate_file(file_path)
        if not is_valid:
            raise FileProcessingError(f"File validation failed: {validation_message}", file_path=file_path, file_type=Path(file_path).suffix.lower())
        try:
            with open(file_path, 'rb') as file:
                files = {'files[]': (Path(file_path).name, file, self._get_mime_type(file_path))}
                data = {
                    'agent_type': agent_type,
                    'model': model,
                    'storage_type': storage_type,
                    'use_finetuning': 'false',
                }
                data.update(options)
                response = self.http_client.post(INITIALIZE_RAG, data=data, files=files)
                if self.logger:
                    self.logger.info("File uploaded successfully", file_path=file_path, response_status=response.get('status', 'unknown'))
                return response
        except Exception as e:
            error_msg = f"Failed to upload file: {str(e)}"
            if self.logger:
                self.logger.error("File upload failed", error=e, file_path=file_path)
            raise FileProcessingError(error_msg, file_path=file_path, file_type=Path(file_path).suffix.lower(), file_size=f"{Path(file_path).stat().st_size / 1024:.1f}KB")

    @require_feature("file_upload")
    def upload_multiple_files(
        self,
        file_paths: List[str],
        agent_type: str = DEFAULT_AGENT_TYPE,
        model: str = DEFAULT_MODEL,
        storage_type: str = 'saved_files',
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Upload multiple files for RAG processing.
        Defaults:
            agent_type: 'general'
            model: platform-specific (Windows: 'unsloth/Qwen3-1.7B-bnb-4bit', Mac: 'mlx-community/Qwen3-4B-4bit')
        """
        if options is None:
            options = {}
        validation_errors = []
        for file_path in file_paths:
            is_valid, validation_message = self.validate_file(file_path)
            if not is_valid:
                validation_errors.append(f"{file_path}: {validation_message}")
        if validation_errors:
            raise FileProcessingError(f"File validation failed for {len(validation_errors)} files: {'; '.join(validation_errors)}", file_path=f"{len(file_paths)} files")
        try:
            files = []
            file_objs = []
            for file_path in file_paths:
                f = open(file_path, 'rb')
                file_objs.append(f)
                files.append(('files[]', (Path(file_path).name, f, self._get_mime_type(file_path))))
            data = {
                'agent_type': agent_type,
                'model': model,
                'storage_type': storage_type,
                'use_finetuning': 'false',
                'file_count': str(len(file_paths))
            }
            data.update(options)
            response = self.http_client.post(INITIALIZE_RAG, data=data, files=files)
            for f in file_objs:
                f.close()
            if self.logger:
                self.logger.info("Multiple files uploaded successfully", file_count=len(file_paths), response_status=response.get('status', 'unknown'))
            return response
        except Exception as e:
            error_msg = f"Failed to upload files: {str(e)}"
            if self.logger:
                self.logger.error("Multiple file upload failed", error=e, file_count=len(file_paths))
            raise FileProcessingError(error_msg, file_path=f"{len(file_paths)} files")

    def get_supported_extensions(self) -> List[str]:
        """Return the list of allowed file extensions."""
        return sorted(ALLOWED_EXTENSIONS)
 