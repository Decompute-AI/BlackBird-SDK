# Confluence Integration API Documentation

This document provides details on the API endpoints for integrating with Confluence. All endpoints require authentication via an Atlassian email, API token, and domain, which must be passed in the request body.

**Base URL:** `http://127.0.0.1:5012`

---

## 1. Test API Status

A simple endpoint to verify that the Confluence blueprint is running.

- **Method:** `GET`
- **URL:** `/api/confluence/test`

### Request

No body or parameters required.

### Response

**On Success (200 OK):**
```json
{
  "status": "success",
  "message": "Confluence blueprint is working!",
  "timestamp": "YYYY-MM-DD HH:MM:SS",
  "endpoints": [
    "POST /api/confluence/validate",
    "POST /api/confluence/test-connection",
    "POST /api/confluence/pages",
    "POST /api/confluence/page/<page_id>",
    "POST /api/confluence/page/<page_id>/images",
    "POST /api/confluence/import-pages"
  ]
}
```

---

## 2. Validate Credentials

Validates a user's Atlassian email, API token, and domain. It also fetches basic user information and a list of accessible workspaces (spaces).

- **Method:** `POST`
- **URL:** `/api/confluence/validate`

### Request Body

```json
{
  "email": "user@company.com",
  "apiToken": "your_atlassian_api_token",
  "domain": "company.atlassian.net"
}
```
*Note: If `domain` is omitted, the backend will attempt to auto-discover it based on the email.*

### Responses

**On Success (200 OK):**
```json
{
  "valid": true,
  "userInfo": { /* User object schema... */ },
  "workspaces": [ /* Array of Space objects schema... */ ],
  "domain": "company.atlassian.net"
}
```

**On Failure (401 Unauthorized):**
```json
{
  "valid": false,
  "error": "Invalid credentials or insufficient permissions",
  "error_code": "INVALID_CREDENTIALS"
}
```

**Other Errors:**
- `400 BAD REQUEST`: If email/token is missing (`MISSING_CREDENTIALS`) or the email format is invalid (`INVALID_EMAIL`).
- `500 INTERNAL SERVER ERROR`: For network issues (`NETWORK_ERROR`) or other backend failures.

---

## 3. Test Connection

A lightweight endpoint to quickly verify that existing credentials are still valid.

- **Method:** `POST`
- **URL:** `/api/confluence/test-connection`

### Request Body

```json
{
  "email": "user@company.com",
  "apiToken": "your_atlassian_api_token",
  "domain": "company.atlassian.net"
}
```

### Responses

**On Success (200 OK):**
```json
{
  "valid": true,
  "userInfo": { /* User object schema... */ }
}
```

**On Failure (401 Unauthorized):**
```json
{
  "valid": false,
  "error": "Connection failed - invalid credentials or domain"
}
```

---

## 4. Get Pages

Fetches a list of pages from Confluence. Can be filtered by a specific space.

- **Method:** `POST`
- **URL:** `/api/confluence/pages`

### Request Body

```json
{
  "email": "user@company.com",
  "apiToken": "your_atlassian_api_token",
  "domain": "company.atlassian.net",
  "spaceKey": "SPACEKEY",
  "limit": 50
}
```
- `spaceKey` (optional): The key of the Confluence space to fetch pages from. If omitted, pages from all accessible spaces are returned.
- `limit` (optional): The number of pages to return. Defaults to 50.

### Response

**On Success (200 OK):**
```json
{
  "valid": true,
  "pages": [ /* Array of Page objects schema... */ ],
  "count": 42
}
```

---

## 5. Get Page Content

Retrieves the full content of a single Confluence page by its ID.

- **Method:** `POST`
- **URL:** `/api/confluence/page/<page_id>`

### URL Parameters
- `page_id`: The ID of the Confluence page.

### Request Body

```json
{
  "email": "user@company.com",
  "apiToken": "your_atlassian_api_token",
  "domain": "company.atlassian.net"
}
```

### Responses

**On Success (200 OK):**
```json
{
  "valid": true,
  "page": { /* Full Page object with body content schema... */ }
}
```

**On Failure (404 Not Found):**
```json
{
  "valid": false,
  "error": "Page not found or access denied"
}
```

---

## 6. Get Page Images

Retrieves all image attachments (e.g., PNG, JPEG, GIF) for a specific Confluence page.

- **Method:** `POST`
- **URL:** `/api/confluence/page/<page_id>/images`

### URL Parameters
- `page_id`: The ID of the Confluence page.

### Request Body

```json
{
  "email": "user@company.com",
  "apiToken": "your_atlassian_api_token",
  "domain": "company.atlassian.net",
  "limit": 50
}
```
- `limit` (optional): The number of image attachments to return. Defaults to 50.

### Response

**On Success (200 OK):**
```json
{
  "valid": true,
  "images": [ /* Array of Attachment objects schema... */ ],
  "count": 3
}
```
*The `images` array will be empty if the page has no image attachments.*

**On Failure (404 Not Found):**
This can occur if the page itself doesn't exist. The response will be similar to the `get_page_content` endpoint.
```json
{
  "valid": false,
  "error": "Page not found or access denied" 
}
```

---

## 7. Import Pages

Imports selected Confluence pages and their image attachments into a Vision Chat knowledge base. This endpoint processes both the text content and images from the specified pages, making them available for AI-powered chat interactions.

- **Method:** `POST`
- **URL:** `/api/confluence/import-pages`

### Request Body

```json
{
  "email": "user@company.com",
  "apiToken": "your_atlassian_api_token",
  "domain": "company.atlassian.net",
  "pageIds": ["12345", "67890"],
  "agent_id": "research",
  "session_id": "uuid-string"
}
```

**Parameters:**
- `email` (required): Atlassian account email
- `apiToken` (required): Atlassian API token
- `domain` (required): Confluence domain (e.g., "company" for company.atlassian.net)
- `pageIds` (required): Array of Confluence page IDs to import
- `agent_id` (optional): Agent identifier for the knowledge base. Defaults to "general"
- `session_id` (optional): Session identifier for the knowledge base. If not provided, a new UUID will be generated

### Processing Details

For each page, the endpoint will:
1. **Extract text content**: Converts HTML content to plain text and adds it to the knowledge base
2. **Download image attachments**: Retrieves all image files (PNG, JPEG, GIF, etc.) attached to the page
3. **Process images**: Adds images to the knowledge base for visual analysis and chat interactions
4. **Clean up**: Removes temporary files after processing

### Responses

**On Success (200 OK):**
```json
{
  "status": "success",
  "session_id": "908dbcf1-6c30-4807-ab53-112fec761fd9"
}
```

**On Failure (400 Bad Request):**
```json
{
  "status": "error",
  "error": "Required: email, apiToken, domain, pageIds"
}
```

**On Failure (500 Internal Server Error):**
```json
{
  "status": "error",
  "error": "Detailed error message"
}
```

### Usage Notes

- The returned `session_id` can be used in subsequent Vision Chat API calls to reference the imported knowledge base
- Images are downloaded using the Confluence Cloud download URLs with proper authentication
- Text content is extracted from HTML using BeautifulSoup when available, falling back to regex-based extraction
- Temporary files are automatically cleaned up after processing
- The endpoint handles missing or inaccessible pages gracefully by skipping them and continuing with the remaining pages 