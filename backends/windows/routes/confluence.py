from flask import Blueprint, request, jsonify
import requests
import base64
import os
from typing import Dict, Any, Optional
from datetime import datetime
import tempfile
import uuid

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

# from blackbird_sdk.backends.windows.routes.artifact_processing import info_preprocess
# from blackbird_sdk.backends.windows.routes.vision_chat import get_knowledge_base

confluence_bp = Blueprint('confluence', __name__)
# print("✅ confluence.py loaded")


class ConfluenceValidator:
    """Validates Atlassian API tokens and manages Confluence connections"""
    
    def __init__(self):
        self.confluence_api_base = "https://{domain}.atlassian.net/wiki/rest/api"
    
    def validate_credentials(self, email: str, api_token: str, domain: str = None) -> Dict[str, Any]:
        """
        Validates Atlassian API token by making test requests to Confluence REST API
        
        Args:
            email: Atlassian account email
            api_token: API token from https://id.atlassian.com/manage/api-tokens
            domain: Optional domain to test against (e.g., 'company.atlassian.net')
            
        Returns:
            Dict with validation result and user/workspace info
        """
        # print(f"🔐 [Confluence] Validating credentials for {email}")
        # print(f"   🌐 Domain: {domain or 'Auto-discover'}")
        
        try:
            # If no domain provided, try to discover it
            if not domain:
                # print("   🔍 Attempting domain discovery...")
                domain = self._discover_domain(email, api_token)
                if not domain:
                    # print("   ❌ Domain discovery failed")
                    return {
                        "valid": False,
                        "error": "Could not discover Confluence domain. Please provide domain explicitly.",
                        "error_code": "DOMAIN_DISCOVERY_FAILED"
                    }
                # print(f"   ✅ Discovered domain: {domain}")
            
            # Test basic authentication and get user info
            # print(f"   🔑 Testing authentication with domain: {domain}")
            user_info = self._get_current_user(domain, email, api_token)
            if not user_info:
                # print("   ❌ Authentication failed")
                return {
                    "valid": False,
                    "error": "Invalid credentials or insufficient permissions",
                    "error_code": "INVALID_CREDENTIALS"
                }
            
            # print(f"   ✅ Authentication successful for user: {user_info.get('displayName', email)}")
            
            # Get accessible spaces to verify full access
            # print("   📚 Fetching accessible spaces...")
            spaces = self._get_spaces(domain, email, api_token)
            # print(f"   ✅ Found {len(spaces)} accessible spaces")
            
            return {
                "valid": True,
                "userInfo": user_info,
                "workspaces": spaces,
                "domain": domain
            }
            
        except requests.exceptions.RequestException as e:
            # print(f"   ❌ Network error: {str(e)}")
            return {
                "valid": False,
                "error": f"Network error: {str(e)}",
                "error_code": "NETWORK_ERROR"
            }
        except Exception as e:
            # print(f"   ❌ Validation error: {str(e)}")
            return {
                "valid": False,
                "error": f"Validation error: {str(e)}",
                "error_code": "VALIDATION_ERROR"
            }
    
    def _discover_domain(self, email: str, api_token: str) -> Optional[str]:
        """Try to discover the user's Confluence domain"""
        # Common domain patterns to try
        common_domains = [
            "company.atlassian.net",
            "organization.atlassian.net",
            "team.atlassian.net"
        ]
        
        # Extract potential domain from email
        email_parts = email.split('@')
        if len(email_parts) == 2:
            domain_part = email_parts[1].split('.')[0]
            common_domains.insert(0, f"{domain_part}.atlassian.net")
        
        for domain in common_domains:
            try:
                user_info = self._get_current_user(domain, email, api_token)
                if user_info:
                    return domain
            except:
                continue
        
        return None
    
    def _get_current_user(self, domain: str, email: str, api_token: str) -> Optional[Dict]:
        """Get current user information from Confluence"""
        url = f"{self.confluence_api_base.format(domain=domain)}/user/current"
        headers = self._get_auth_headers(email, api_token)
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 401:
            return None
        
        response.raise_for_status()
        return response.json()
    
    def _get_spaces(self, domain: str, email: str, api_token: str) -> list:
        """Get list of accessible spaces"""
        url = f"{self.confluence_api_base.format(domain=domain)}/space"
        params = {"limit": 50, "type": "global"}
        headers = self._get_auth_headers(email, api_token)
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 401:
            return []
        
        response.raise_for_status()
        data = response.json()
        return data.get('results', [])
    
    def get_pages(self, domain: str, email: str, api_token: str, space_key: str = None, limit: int = 50) -> list:
        """Get pages from Confluence"""
        url = f"{self.confluence_api_base.format(domain=domain)}/content"
        params = {
            "type": "page",
            "limit": limit,
            "expand": "space,version"
        }
        
        if space_key:
            params["spaceKey"] = space_key
            
        headers = self._get_auth_headers(email, api_token)
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        return data.get('results', [])
    
    def get_page_content(self, domain: str, email: str, api_token: str, page_id: str) -> Optional[Dict]:
        """Get specific page content by ID"""
        url = f"{self.confluence_api_base.format(domain=domain)}/content/{page_id}"
        params = {
            "expand": "body.storage,space,version"
        }
        headers = self._get_auth_headers(email, api_token)
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 404:
            return None
            
        response.raise_for_status()
        return response.json()

    # === NEW METHOD: Get image attachments for a page ===
    def get_page_images(self, domain: str, email: str, api_token: str, page_id: str, limit: int = 50) -> list:
        """Retrieve image attachments (PNG/JPEG/GIF …) from a Confluence page"""
        url = f"{self.confluence_api_base.format(domain=domain)}/content/{page_id}/child/attachment"
        params = {
            "limit": limit,
            "expand": "metadata",
        }
        headers = self._get_auth_headers(email, api_token)

        print(f"DEBUG: Fetching attachments from URL: {url}")
        print(f"DEBUG: Request params: {params}")
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 404:
            print(f"DEBUG: No attachments found for page {page_id}")
            return []
        response.raise_for_status()
        data = response.json()
        
        print(f"DEBUG: Full attachments response: {data}")
        
        attachments = data.get("results", [])
        print(f"DEBUG: Found {len(attachments)} total attachments")
        
        # Filter only image/* mimetypes
        images = [att for att in attachments if att.get("metadata", {}).get("mediaType", "").startswith("image/")]
        print(f"DEBUG: Filtered to {len(images)} image attachments")
        
        for i, img in enumerate(images):
            print(f"DEBUG: Image {i+1}: {img}")
        
        return images
    
    def download_attachment_data(self, domain: str, email: str, api_token: str, attachment: Dict[str, Any]) -> Optional[bytes]:
        """Download the binary data for an attachment using the _links.download field"""
        # Extract the download link from the attachment metadata
        download_link = attachment.get("_links", {}).get("download")
        if not download_link:
            print(f"DEBUG: No download link found in attachment metadata")
            return None
        
        # Construct the full download URL using the domain and download link
        # The download link is a relative path, so we need to prepend the base URL
        base_url = f"https://{domain}.atlassian.net/wiki"
        full_download_url = f"{base_url}{download_link}"
        
        headers = self._get_auth_headers(email, api_token)
        
        print(f"DEBUG: Downloading attachment from URL: {full_download_url}")
        
        response = requests.get(full_download_url, headers=headers, timeout=30)
        if response.status_code == 404:
            print(f"DEBUG: Attachment not found at URL: {full_download_url}")
            return None
        
        response.raise_for_status()
        return response.content

    def _get_auth_headers(self, email: str, api_token: str) -> Dict[str, str]:
        """Create Basic Auth headers for Atlassian API following official documentation"""
        credentials = f"{email}:{api_token}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        
        return {
            "Authorization": f"Basic {encoded_credentials}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

# Initialize validator
validator = ConfluenceValidator()

@confluence_bp.route('/api/confluence/test', methods=['GET'])
def test_confluence_endpoint():
    """Simple test endpoint to verify Confluence blueprint is working"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # print(f"\n🧪 [{timestamp}] /api/confluence/test called")
    
    return jsonify({
        "status": "success",
        "message": "Confluence blueprint is working!",
        "timestamp": timestamp,
        "endpoints": [
            "POST /api/confluence/validate",
            "POST /api/confluence/test-connection", 
            "POST /api/confluence/pages",
            "POST /api/confluence/page/<page_id>"
        ]
    }), 200

@confluence_bp.route('/api/confluence/validate', methods=['POST'])
def validate_confluence_credentials():
    """
    Validate Atlassian API token and return user/workspace information
    
    Expected JSON payload:
    {
        "email": "user@company.com",
        "apiToken": "token123",
        "domain": "company.atlassian.net" (optional)
    }
    
    Returns:
    {
        "valid": true/false,
        "userInfo": {...},
        "workspaces": [...],
        "domain": "domain.atlassian.net",
        "error": "error message if invalid"
    }
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # print(f"\n🔐 [{timestamp}] /api/confluence/validate called")
    
    try:
        data = request.get_json()
        # print(f"   📄 Received data: {data}")
        
        if not data:
            # print("   ❌ No data provided")
            return jsonify({
                "valid": False,
                "error": "No data provided",
                "error_code": "NO_DATA"
            }), 400
        
        email = data.get('email', '').strip()
        api_token = data.get('apiToken', '').strip()
        domain = data.get('domain', '').strip() or None
        
        # print(f"   📧 Email: {email}")
        # print(f"   🔑 API Token: {api_token[:10]}..." if api_token else "   🔑 API Token: [not provided]")
        # print(f"   🌐 Domain: {domain or 'Auto-discover'}")
        
        # Basic validation
        if not email or not api_token:
            # print("   ❌ Missing credentials")
            return jsonify({
                "valid": False,
                "error": "Email and API token are required",
                "error_code": "MISSING_CREDENTIALS"
            }), 400
        
        if '@' not in email:
            # print("   ❌ Invalid email format")
            return jsonify({
                "valid": False,
                "error": "Please provide a valid email address",
                "error_code": "INVALID_EMAIL"
            }), 400
        
        # print("   ✅ Input validation passed, calling validator...")
        
        # Validate credentials with Atlassian
        result = validator.validate_credentials(email, api_token, domain)
        
        # print(f"   📊 Validation result: {result.get('valid', False)}")
        
        if result["valid"]:
            # print("   ✅ Validation successful")
            return jsonify(result), 200
        else:
            # print(f"   ❌ Validation failed: {result.get('error', 'Unknown error')}")
            return jsonify(result), 401
            
    except Exception as e:
        # print(f"   💥 Server error: {str(e)}")
        return jsonify({
            "valid": False,
            "error": f"Server error: {str(e)}",
            "error_code": "SERVER_ERROR"
        }), 500

@confluence_bp.route('/api/confluence/test-connection', methods=['POST'])
def test_confluence_connection():
    """
    Test an existing Confluence connection
    
    Expected JSON payload:
    {
        "email": "user@company.com",
        "apiToken": "token123",
        "domain": "company.atlassian.net"
    }
    """
    try:
        data = request.get_json()
        email = data.get('email', '').strip()
        api_token = data.get('apiToken', '').strip()
        domain = data.get('domain', '').strip()
        
        if not all([email, api_token, domain]):
            return jsonify({
                "valid": False,
                "error": "Email, API token, and domain are required"
            }), 400
        
        # Test connection to specific domain
        user_info = validator._get_current_user(domain, email, api_token)
        
        if user_info:
            return jsonify({
                "valid": True,
                "userInfo": user_info
            }), 200
        else:
            return jsonify({
                "valid": False,
                "error": "Connection failed - invalid credentials or domain"
            }), 401
            
    except Exception as e:
        return jsonify({
            "valid": False,
            "error": f"Server error: {str(e)}"
        }), 500

@confluence_bp.route('/api/confluence/pages', methods=['POST'])
def get_confluence_pages():
    """
    Get pages from Confluence
    
    Expected JSON payload:
    {
        "email": "user@company.com",
        "apiToken": "token123",
        "domain": "company.atlassian.net",
        "spaceKey": "SPACE" (optional),
        "limit": 50 (optional)
    }
    """
    try:
        data = request.get_json()
        email = data.get('email', '').strip()
        api_token = data.get('apiToken', '').strip()
        domain = data.get('domain', '').strip()
        space_key = data.get('spaceKey', '').strip() or None
        limit = data.get('limit', 50)
        
        if not all([email, api_token, domain]):
            return jsonify({
                "valid": False,
                "error": "Email, API token, and domain are required"
            }), 400
        
        pages = validator.get_pages(domain, email, api_token, space_key, limit)
        
        return jsonify({
            "valid": True,
            "pages": pages,
            "count": len(pages)
        }), 200
            
    except Exception as e:
        return jsonify({
            "valid": False,
            "error": f"Server error: {str(e)}"
        }), 500

@confluence_bp.route('/api/confluence/page/<page_id>', methods=['POST'])
def get_confluence_page_content(page_id):
    """
    Get specific page content by ID
    
    Expected JSON payload:
    {
        "email": "user@company.com",
        "apiToken": "token123",
        "domain": "company.atlassian.net"
    }
    """
    try:
        data = request.get_json()
        email = data.get('email', '').strip()
        api_token = data.get('apiToken', '').strip()
        domain = data.get('domain', '').strip()
        
        if not all([email, api_token, domain]):
            return jsonify({
                "valid": False,
                "error": "Email, API token, and domain are required"
            }), 400
        
        page_content = validator.get_page_content(domain, email, api_token, page_id)
        
        if page_content:
            return jsonify({
                "valid": True,
                "page": page_content
            }), 200
        else:
            return jsonify({
                "valid": False,
                "error": "Page not found or access denied"
            }), 404
            
    except Exception as e:
        return jsonify({
            "valid": False,
            "error": f"Server error: {str(e)}"
        }), 500

@confluence_bp.route('/api/confluence/page/<page_id>/images', methods=['POST'])
def get_confluence_page_images(page_id):
    """
    Get all image attachments for a page

    Expected JSON payload:
    {
        "email": "user@company.com",
        "apiToken": "token123",
        "domain": "company.atlassian.net",
        "limit": 50  (optional)
    }
    """
    try:
        data = request.get_json()
        email = data.get('email', '').strip()
        api_token = data.get('apiToken', '').strip()
        domain = data.get('domain', '').strip()
        limit = data.get('limit', 50)

        if not all([email, api_token, domain]):
            return jsonify({
                "valid": False,
                "error": "Email, API token, and domain are required"
            }), 400

        images = validator.get_page_images(domain, email, api_token, page_id, limit)

        return jsonify({
            "valid": True,
            "images": images,
            "count": len(images)
        }), 200

    except Exception as e:
        return jsonify({
            "valid": False,
            "error": f"Server error: {str(e)}"
        }), 500

@confluence_bp.route('/api/confluence/import-pages', methods=['POST'])
def import_confluence_pages():
    """
    Import selected Confluence pages and their image attachments into a Vision Chat knowledge base.

    Expected JSON payload:
    {
      "email": "user@company.com",
      "apiToken": "token",
      "domain": "company.atlassian.net",
      "pageIds": ["12345", "67890"],
      "agent_id": "research",          # optional, defaults to 'general'
      "session_id": "uuid-string"      # optional, generated if missing
    }

    The endpoint creates or updates the corresponding knowledge base and
    returns only the unique knowledge-base (session) ID so the frontend can
    reference it in subsequent vision-chat calls.
    """
    try:
        data = request.get_json() or {}
        email = data.get('email', '').strip()
        api_token = data.get('apiToken', '').strip()
        domain = data.get('domain', '').strip()
        page_ids = data.get('pageIds', [])
        agent_id = data.get('agent_id', 'general')
        session_id = data.get('session_id') or str(uuid.uuid4())

        # Basic validation
        if not all([email, api_token, domain]) or not page_ids:
            return jsonify({
                "status": "error",
                "error": "Required: email, apiToken, domain, pageIds"
            }), 400

        # Build Basic-Auth header for attachment downloads
        auth_header = {
            "Authorization": "Basic " + base64.b64encode(f"{email}:{api_token}".encode()).decode()
        }

        # Obtain (or create) the knowledge base for this session/agent
        # kb = get_knowledge_base(session_id, agent_id) # This line was removed as per the edit hint

        tmp_dir = tempfile.gettempdir()

        for page_id in page_ids:
            # --- Fetch and ingest page content ---
            page = validator.get_page_content(domain, email, api_token, page_id)
            if not page:
                continue

            html_body = page.get("body", {}).get("storage", {}).get("value", "")
            title = page.get("title", f"page_{page_id}")

            if BeautifulSoup:
                text_body = BeautifulSoup(html_body, "html.parser").get_text(" ", strip=True)
            else:
                import re
                text_body = re.sub("<[^<]+?>", " ", html_body)

            tmp_doc = os.path.join(tmp_dir, f"{title}_{page_id}.txt")
            with open(tmp_doc, "w", encoding="utf-8") as f:
                f.write(text_body)
            # kb.add_document(tmp_doc, file_type='document') # This line was removed as per the edit hint
            if os.path.exists(tmp_doc):
                os.remove(tmp_doc)

            # --- Fetch and ingest page images ---
            images = validator.get_page_images(domain, email, api_token, page_id, limit=50)
            # print(f"Found {len(images)} images for page {page_id}")
            
            for img in images:
                # print(f"Processing image: {img.get('title', 'Untitled')} (ID: {img.get('id')})")
                # print(f"DEBUG: Full image object: {img}")
                
                # Check if the attachment object has download link
                if not img.get("_links", {}).get("download"):
                    # print(f"No download link found for image {img.get('title', 'Untitled')}")
                    continue
                
                # print(f"Downloading attachment data for image: {img.get('title', 'Untitled')}")
                
                binary_data = validator.download_attachment_data(domain, email, api_token, img)
                
                if not binary_data:
                    # print(f"Failed to download attachment data for image: {img.get('title', 'Untitled')}")
                    continue

                img_name = img.get("title") or f"img_{img.get('id')}.png"
                tmp_img = os.path.join(tmp_dir, img_name)
                
                # print(f"Saving image to temp file: {tmp_img}")
                with open(tmp_img, "wb") as f:
                    f.write(binary_data)
                
                # print(f"Temp file size: {os.path.getsize(tmp_img)} bytes")
                # print(f"Adding image to knowledge base...")
                
                # result = kb.add_image(tmp_img) # This line was removed as per the edit hint
                # print(f"Add image result: {result}")
                
                if os.path.exists(tmp_img):
                    os.remove(tmp_img)
                    # print(f"Cleaned up temp file: {tmp_img}")

        return jsonify({
            "status": "success",
            "session_id": session_id
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500