"""Enhanced HTTP client with comprehensive error handling and retry mechanisms."""

import requests
import time
import json
import logging
from urllib.parse import urljoin
from typing import Dict, Any, Optional
from .errors import *

class HTTPClient:
    """Enhanced client for making HTTP requests with robust error handling."""
    
    def __init__(self, config, logger=None):
        """Initialize the HTTP client with enhanced error handling."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.session = requests.Session()
        self.base_url = config.get('base_url')
        self.timeout = config.get('timeout', 120000) / 1000
        
        # Enhanced retry configuration
        self.max_retries = config.get('retry_count', 3)
        self.retry_delay = config.get('retry_delay', 1000) / 1000
        self.retry_backoff = config.get('retry_backoff', 2.0)
        
        # Configure connection pooling for streaming
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )
        
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # Add default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'DecomputeSDK/0.1.0',
            'Connection': 'keep-alive'
        })
        
    def _prepare_headers(self, headers=None):
        """Prepare request headers with authentication if available."""
        default_headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        }
        
        api_key = self.config.get('api_key')
        if api_key:
            default_headers['Authorization'] = f'Bearer {api_key}'
            
        if headers:
            default_headers.update(headers)
            
        return default_headers
    
    def _should_retry(self, exception, attempt, endpoint):
        """Determine if a request should be retried based on the error type."""
        if attempt >= self.max_retries:
            return False
        
        # Retry on network errors
        if isinstance(exception, (requests.exceptions.ConnectionError, 
                                requests.exceptions.Timeout,
                                requests.exceptions.ReadTimeout)):
            self.logger.warning(f"Network error on {endpoint}, attempt {attempt + 1}: {exception}")
            return True
        
        # Retry on server errors (5xx)
        if isinstance(exception, requests.exceptions.HTTPError):
            if 500 <= exception.response.status_code < 600:
                self.logger.warning(f"Server error {exception.response.status_code} on {endpoint}, attempt {attempt + 1}")
                return True
        
        return False
    
    def _calculate_retry_delay(self, attempt):
        """Calculate retry delay with exponential backoff."""
        delay = self.retry_delay * (self.retry_backoff ** attempt)
        return min(delay, 30)  # Cap at 30 seconds
    
    def _parse_streaming_response(self, response, endpoint="unknown"):
        """Enhanced streaming response parser with error handling."""
        try:
            content_type = response.headers.get('content-type', '')
            
            # Handle SSE (Server-Sent Events) format
            if 'text/event-stream' in content_type:
                return self._parse_sse_response(response)
            
            # Handle regular streaming response
            chunks = []
            for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                if chunk:
                    chunks.append(chunk)
            
            full_content = ''.join(chunks) if isinstance(chunks[0], str) else b''.join(chunks)
            
            try:
                # Try to parse as JSON
                if isinstance(full_content, str):
                    return json.loads(full_content)
                else:
                    return json.loads(full_content.decode('utf-8'))
            except (ValueError, UnicodeDecodeError):
                # Return raw content if not JSON
                return {'content': full_content}
                
        except Exception as e:
            self.logger.error(f"Error parsing streaming response: {e}")
            raise StreamingResponseError(
                f"Failed to parse streaming response: {str(e)}",
                response_format="parse_error"
            )
    
    def _parse_sse_response(self, response):
        """Parse Server-Sent Events (SSE) formatted response."""
        chunks = []
        events = []
        
        try:
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    if line.startswith('data:'):
                        data = line[5:].strip()
                        chunks.append(data)
                        
                        # Try to parse as JSON
                        try:
                            event_data = json.loads(data)
                            events.append(event_data)
                        except json.JSONDecodeError:
                            events.append(data)
            
            return {
                'status': 'success',
                'events': events,
                'raw_data': chunks
            }
        except Exception as e:
            self.logger.error(f"Error parsing SSE response: {e}")
            raise StreamingResponseError(
                f"Failed to parse SSE response: {str(e)}",
                response_format="sse_parse_error"
            )
    
    def _handle_response(self, response, endpoint):
        """Enhanced response handling with detailed error categorization."""
        try:
            response.raise_for_status()
            
            if not response.content:
                return {'status': 'success', 'message': 'Request completed successfully'}
            
            content_type = response.headers.get('content-type', '')
            
            # Handle streaming responses
            if 'text/event-stream' in content_type or getattr(response, 'streaming', False):
                return self._parse_streaming_response(response, endpoint)
            else:
                try:
                    return response.json()
                except ValueError as e:
                    self.logger.warning(f"Failed to parse JSON response from {endpoint}: {e}")
                    return {'content': response.content.decode('utf-8')}
                    
        except requests.exceptions.HTTPError as e:
            status_code = response.status_code
            
            # Enhanced error categorization
            if status_code == 401:
                raise AuthenticationError(
                    "Authentication failed. Check your API key.",
                    api_key_status="invalid" if self.config.get('api_key') else "missing"
                )
            elif status_code == 403:
                raise AuthenticationError(
                    "Access forbidden. Check your permissions.",
                    api_key_status="insufficient_permissions"
                )
            elif status_code == 404:
                raise APIError(
                    f"Endpoint not found: {endpoint}",
                    status_code=status_code,
                    endpoint=endpoint
                )
            elif status_code == 405:
                raise APIError(
                    f"Method not allowed for endpoint: {endpoint}",
                    status_code=status_code,
                    endpoint=endpoint
                )
            elif status_code == 429:
                raise APIError(
                    "Rate limit exceeded. Please wait before making more requests.",
                    status_code=status_code,
                    endpoint=endpoint
                )
            elif 500 <= status_code < 600:
                raise APIError(
                    f"Server error: {status_code}",
                    status_code=status_code,
                    endpoint=endpoint
                )
            else:
                # Try to extract error message from response
                error_message = "API Error"
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_message = error_data['error']
                    elif 'message' in error_data:
                        error_message = error_data['message']
                except ValueError:
                    if response.content:
                        error_message = response.content.decode('utf-8')[:200]
                
                raise APIError(
                    error_message,
                    status_code=status_code,
                    response=error_data if 'error_data' in locals() else None,
                    endpoint=endpoint
                )
    
    def _retry_with_recovery(self, operation, *args, **kwargs):
        """Execute operation with retry logic and error recovery."""
        last_exception = None
        endpoint = kwargs.pop('endpoint', 'unknown')
        
        for attempt in range(self.max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if not self._should_retry(e, attempt, endpoint):
                    break
                
                if attempt < self.max_retries:
                    delay = self._calculate_retry_delay(attempt)
                    self.logger.info(f"Retrying in {delay:.2f}s... (attempt {attempt + 1}/{self.max_retries + 1})")
                    time.sleep(delay)
        
        # If we get here, all retries failed
        raise NetworkError(
            f"Request failed after {self.max_retries} retries: {str(last_exception)}",
            retry_count=self.max_retries,
            endpoint=endpoint
        )
    
    def _make_streaming_request(self, method, endpoint, **kwargs):
        """Enhanced streaming request with proper error handling."""
        try:
            # Ensure stream=True is properly set
            kwargs['stream'] = True
            kwargs['timeout'] = (30, 300)  # (connect, read) timeout
            
            url = urljoin(self.base_url, endpoint.lstrip('/'))
            headers = kwargs.get('headers', {})
            
            # Add Accept header for SSE if not present
            if 'Accept' not in headers:
                headers['Accept'] = 'text/event-stream, application/json'
            
            kwargs['headers'] = headers
            
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            
            return response
            
        except requests.exceptions.Timeout as e:
            raise NetworkError(f"Request timed out: {str(e)}")
        except requests.exceptions.ConnectionError as e:
            raise NetworkError(f"Connection error: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Request failed: {str(e)}")
    
    def process_streaming_response(self, response):
        """Process streaming response with proper chunk handling."""
        try:
            for chunk in response.iter_content(chunk_size=8192, decode_unicode=False):
                if chunk:  # Filter out keep-alive chunks
                    yield chunk
        except Exception as e:
            response.close()  # Ensure connection cleanup
            raise NetworkError(f"Error in streaming response: {str(e)}")
        finally:
            response.close()
    
    def request(self, method, endpoint, data=None, params=None, headers=None, files=None, stream=False, **kwargs):
        """Make a request with comprehensive error handling and retry logic."""
        url = urljoin(self.base_url, endpoint.lstrip('/'))
        headers = self._prepare_headers(headers)
        
        self.logger.debug(f"Making {method} request to: {url}")
        
        # Set appropriate timeouts based on endpoint
        if endpoint == '/chat':
            kwargs['timeout'] = (30, 300)  # 5 minutes for AI inference
        elif '/initialize' in endpoint:
            kwargs['timeout'] = (30, 180)  # 3 minutes for initialization
        else:
            kwargs['timeout'] = (15, 60)   # 1 minute for other operations
        
        if data and self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Request data: {json.dumps(data)[:200]}")
        
        def _make_request():
            try:
                if stream:
                    return self._make_streaming_request(method, endpoint, 
                                                      json=data, 
                                                      params=params, 
                                                      headers=headers)
                
                if files:
                    headers_copy = headers.copy()
                    headers_copy.pop('Content-Type', None)
                    response = self.session.request(
                        method=method,
                        url=url,
                        files=files,
                        data=data,
                        params=params,
                        headers=headers_copy,
                        **kwargs
                    )
                else:
                    response = self.session.request(
                        method=method,
                        url=url,
                        json=data,
                        params=params,
                        headers=headers,
                        stream=stream,
                        **kwargs
                    )
                
                self.logger.debug(f"Response status: {response.status_code}")
                return self._handle_response(response, endpoint)
                
            except requests.exceptions.Timeout as e:
                timeout_val = kwargs.get('timeout', self.timeout)
                raise TimeoutError(
                    f"Request timeout after {timeout_val}s for {endpoint}",
                    timeout_duration=timeout_val,
                    operation=f"{method} {endpoint}"
                )
            except requests.exceptions.ConnectionError as e:
                raise NetworkError(
                    f"Connection error: {str(e)}",
                    endpoint=endpoint
                )
        
        return self._retry_with_recovery(_make_request, endpoint=endpoint)
    
    def get(self, endpoint, params=None, headers=None, stream=False):
        """Make a GET request."""
        return self.request('GET', endpoint, params=params, headers=headers, stream=stream)
    
    def post(self, endpoint, data=None, params=None, headers=None, files=None, stream=False):
        """Enhanced POST method with proper file upload support."""
        url = urljoin(self.base_url, endpoint.lstrip('/'))
        
        # Handle file uploads differently
        if files:
            # Remove Content-Type header for multipart uploads (only for this request)
            if 'Content-Type' in self.session.headers:
                del self.session.headers['Content-Type']
            headers_copy = (headers or {}).copy()
            headers_copy.pop('Content-Type', None)
            
            # Don't send JSON for file uploads - use form data
            try:
                response = self.session.post(
                    url,
                    data=data,  # Send as form data, not JSON
                    files=files,
                    params=params,
                    headers=headers_copy,
                    timeout=self.timeout
                )
                
                self.logger.debug(f"File upload response status: {response.status_code}")
                return self._handle_response(response, endpoint)
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"File upload request failed: {e}")
                raise APIError(f"File upload failed: {str(e)}")
        else:
            # Regular JSON request
            headers = self._prepare_headers(headers)
            
            try:
                if data is not None:
                    data = json.dumps(data)
                
                response = self.session.post(
                    url,
                    data=data,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                    stream=stream
                )
                
                return self._handle_response(response, endpoint)
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"POST request failed: {e}")
                raise APIError(f"Request failed: {str(e)}")

    def put(self, endpoint, data=None, params=None, headers=None):
        """Make a PUT request."""
        return self.request('PUT', endpoint, data=data, params=params, headers=headers)
    
    def delete(self, endpoint, params=None, headers=None):
        """Make a DELETE request."""
        return self.request('DELETE', endpoint, params=params, headers=headers)
    
    def stream_request(self, method, endpoint, data=None, params=None, headers=None):
        """Make a streaming request with enhanced error handling."""
        try:
            # Validate backend connection first
            self.validate_backend_connection()
            
            # Make streaming request
            response = self._make_streaming_request(method, endpoint, 
                                                  json=data, 
                                                  params=params, 
                                                  headers=headers)
            
            return self.process_streaming_response(response)
            
        except Exception as e:
            self.logger.error(f"Streaming request failed: {e}")
            raise
    
    def validate_backend_connection(self):
        """Validate backend is ready for streaming."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code != 200:
                raise NetworkError("Backend not ready")
            return True
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Backend connection failed: {e}")
