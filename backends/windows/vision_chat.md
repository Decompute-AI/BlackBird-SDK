# Vision Chat API Documentation

This document provides details on the API endpoints for the Vision Chat system with knowledge base integration. The Vision Chat system allows AI-powered conversations with documents, images, and other content stored in a knowledge base.

**Base URL:** `http://127.0.0.1:5012`

---

## Overview

The Vision Chat API enables:
- AI-powered conversations with knowledge base content
- Document and image processing with embedding-based retrieval
- Multi-modal chat supporting text, PDFs, and images
- Knowledge base management and content organization
- Advanced search and retrieval capabilities

---

## 1. Initialize Vision Chat Session

Initialize a vision chat session with an existing knowledge base (e.g., from Confluence import).

- **Method:** `POST`
- **URL:** `/vision-chat-initialize`

### Request Body

```json
{
  "agent": "research",
  "model_name": "unsloth/Qwen3-1.7B-bnb-4bit",
  "session_id": "908dbcf1-6c30-4807-ab53-112fec761fd9"
}
```

**Parameters:**
- `agent` (required): Agent type - one of: `general`, `tech`, `legal`, `finance`, `meetings`, `research`, `image-generator`
- `model_name` (optional): Model to use. Defaults to `unsloth/Qwen3-1.7B-bnb-4bit`
- `session_id` (required): Session ID from previous knowledge base creation (e.g., from Confluence import)

### Response

**On Success (200 OK):**
```json
{
  "status": "success",
  "message": "Research vision agent initialized successfully",
  "agent": "research",
  "model": "unsloth/Qwen3-1.7B-bnb-4bit",
  "session_id": "908dbcf1-6c30-4807-ab53-112fec761fd9",
  "knowledge_base": {
    "session_id": "908dbcf1-6c30-4807-ab53-112fec761fd9",
    "agent_id": "research",
    "total_documents": 5,
    "total_images": 12,
    "total_processed": 17,
    "created_at": "2024-01-20T10:30:00",
    "last_updated": "2024-01-20T11:45:00",
    "index_info": {
      "total_chunks": 150,
      "last_indexed": "2024-01-20T11:45:00",
      "index_version": "2.0",
      "embedding_model": "hkunlp/instructor-large"
    }
  }
}
```

**On Error (400/500):**
```json
{
  "error": "Invalid agent type. Must be one of: general, tech, legal, finance, meetings, research, image-generator"
}
```

---

## 2. Vision Chat Conversation

Start a conversation with the AI using the knowledge base context. This endpoint supports streaming responses.

- **Method:** `POST`
- **URL:** `/vision-chat`

### Request Body

```json
{
  "message": "What are the main findings from the research documents?",
  "agent": "research",
  "model": "unsloth/Qwen3-1.7B-bnb-4bit",
  "session_id": "908dbcf1-6c30-4807-ab53-112fec761fd9",
  "include_history": true,
  "clear_history": false
}
```

**Parameters:**
- `message` (required): User's message/question
- `agent` (required): Agent type (must match the initialized agent)
- `model` (required): Model name (must match the initialized model)
- `session_id` (required): Session ID for the knowledge base
- `include_history` (optional): Whether to include conversation history. Defaults to `true`
- `clear_history` (optional): Whether to clear conversation history. Defaults to `false`

### Response

**Streaming Response (Server-Sent Events):**

The response is streamed using Server-Sent Events (SSE). Each chunk contains:

```json
{
  "response": "text chunk",
  "tokens_per_second": 25.5
}
```

**Final Response:**
```json
{
  "response": "complete response text",
  "replace": true
}
```

**Stream End:**
```json
{
  "status": "complete"
}
```

**On Error:**
```json
{
  "error": "Error message",
  "status": "error"
}
```

### JavaScript Example

```javascript
const response = await fetch('/vision-chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    message: "What are the key points from the uploaded documents?",
    agent: "research",
    model: "unsloth/Qwen3-1.7B-bnb-4bit",
    session_id: "908dbcf1-6c30-4807-ab53-112fec761fd9"
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      if (data.response) {
        // Handle streaming response
        console.log(data.response);
      }
    }
  }
}
```

---

## 3. Knowledge Base Management

### 3.1 Get Knowledge Base Information

Retrieve information about the knowledge base content.

- **Method:** `GET`
- **URL:** `/vision-chat/knowledge-base/<session_id>`

### Query Parameters
- `agent_id` (required): Agent identifier

### Response

```json
{
  "session_id": "908dbcf1-6c30-4807-ab53-112fec761fd9",
  "agent_id": "research",
  "total_documents": 5,
  "total_images": 12,
  "total_processed": 17,
  "created_at": "2024-01-20T10:30:00",
  "last_updated": "2024-01-20T11:45:00",
  "index_info": {
    "total_chunks": 150,
    "last_indexed": "2024-01-20T11:45:00",
    "index_version": "2.0",
    "embedding_model": "hkunlp/instructor-large"
  }
}
```

### 3.2 List Knowledge Base Content

Get a detailed list of all documents and images in the knowledge base.

- **Method:** `GET`
- **URL:** `/vision-chat/list-content/<session_id>`

### Query Parameters
- `agent_id` (required): Agent identifier

### Response

```json
{
  "status": "success",
  "session_id": "908dbcf1-6c30-4807-ab53-112fec761fd9",
  "agent_id": "research",
  "documents": [
    {
      "id": "doc-uuid-1",
      "name": "Research_Report.pdf",
      "type": "pdf",
      "added_at": "2024-01-20T10:30:00",
      "pages": 15,
      "path": "/path/to/document"
    }
  ],
  "images": [
    {
      "id": "img-uuid-1",
      "name": "chart.png",
      "added_at": "2024-01-20T10:35:00",
      "description": "Bar chart showing quarterly results",
      "path": "/path/to/image"
    }
  ],
  "total_documents": 5,
  "total_images": 12,
  "total_chunks": 150
}
```

### 3.3 Search Knowledge Base

Search for specific content within the knowledge base using advanced embedding-based retrieval.

- **Method:** `POST`
- **URL:** `/vision-chat/search/<session_id>`

### Request Body

```json
{
  "query": "financial performance metrics",
  "agent_id": "research",
  "max_results": 5
}
```

### Response

```json
{
  "status": "success",
  "query": "financial performance metrics",
  "results": [
    {
      "content_id": "doc-uuid-1",
      "content_type": "pdf_page",
      "best_chunk": "The financial performance metrics show a 15% increase in revenue...",
      "relevance_score": 0.89,
      "metadata": {
        "page_number": 3,
        "document_name": "Q4_Report.pdf",
        "has_images": true
      },
      "all_chunks_count": 3,
      "total_score": 2.45
    }
  ],
  "total_results": 5
}
```

---

## 4. Content Management

### 4.1 Add Document

Add a new document to the knowledge base.

- **Method:** `POST`
- **URL:** `/vision-chat/add-document`

### Request Body (Form Data)

```form-data
session_id: "908dbcf1-6c30-4807-ab53-112fec761fd9"
agent_id: "research"
file: [PDF file]
```

### Response

```json
{
  "status": "success",
  "message": "Document added to knowledge base successfully",
  "document": {
    "doc_id": "new-doc-uuid",
    "name": "New_Document.pdf",
    "processed": true,
    "content": {
      "doc_id": "new-doc-uuid",
      "type": "pdf",
      "pages": 10,
      "summary": "Document summary...",
      "processed_at": "2024-01-20T12:00:00"
    }
  }
}
```

### 4.2 Add Image

Add a new image to the knowledge base.

- **Method:** `POST`
- **URL:** `/vision-chat/add-image`

### Request Body (Form Data)

```form-data
session_id: "908dbcf1-6c30-4807-ab53-112fec761fd9"
agent_id: "research"
file: [Image file]
```

### Response

```json
{
  "status": "success",
  "message": "Image added to knowledge base successfully",
  "image": {
    "img_id": "new-img-uuid",
    "name": "chart.png",
    "processed": true,
    "content": {
      "img_id": "new-img-uuid",
      "type": "image",
      "description": "A bar chart showing quarterly sales data",
      "processed_at": "2024-01-20T12:00:00"
    }
  }
}
```

### 4.3 Delete Content

Delete specific content from the knowledge base.

#### Delete Document
- **Method:** `DELETE`
- **URL:** `/vision-chat/delete-document/<session_id>`

```json
{
  "doc_id": "doc-uuid-to-delete",
  "agent_id": "research"
}
```

#### Delete Image
- **Method:** `DELETE`
- **URL:** `/vision-chat/delete-image/<session_id>`

```json
{
  "img_id": "img-uuid-to-delete",
  "agent_id": "research"
}
```

### 4.4 Clear Knowledge Base

Clear all content from the knowledge base.

- **Method:** `DELETE`
- **URL:** `/vision-chat/clear-kb/<session_id>`

### Request Body

```json
{
  "agent_id": "research"
}
```

### Response

```json
{
  "status": "success",
  "message": "Knowledge base cleared successfully",
  "cleared_at": "2024-01-20T12:00:00"
}
```

---

## 5. Index Management

### 5.1 Get Index Statistics

Get detailed statistics about the embedding index.

- **Method:** `GET`
- **URL:** `/vision-chat/index-stats/<session_id>`

### Query Parameters
- `agent_id` (required): Agent identifier

### Response

```json
{
  "session_id": "908dbcf1-6c30-4807-ab53-112fec761fd9",
  "agent_id": "research",
  "embedding_model_available": true,
  "faiss_index_available": true,
  "embedding_dimension": 768,
  "total_chunks": 150,
  "total_documents": 5,
  "total_images": 12,
  "total_processed": 17,
  "index_file_exists": true,
  "mapping_file_exists": true,
  "metadata_file_exists": true,
  "content_type_distribution": {
    "pdf_page": 120,
    "image": 15,
    "document": 15
  }
}
```

### 5.2 Rebuild Index

Rebuild the embedding index (useful after bulk operations or corruption).

- **Method:** `POST`
- **URL:** `/vision-chat/rebuild-index/<session_id>`

### Request Body

```json
{
  "agent_id": "research"
}
```

### Response

```json
{
  "status": "success",
  "message": "Embedding index rebuilt successfully",
  "total_chunks": 150,
  "index_size": 150
}
```

### 5.3 Health Check

Check the health and integrity of the knowledge base.

- **Method:** `GET`
- **URL:** `/vision-chat/health-check/<session_id>`

### Query Parameters
- `agent_id` (required): Agent identifier

### Response

```json
{
  "session_id": "908dbcf1-6c30-4807-ab53-112fec761fd9",
  "agent_id": "research",
  "status": "healthy",
  "issues": [],
  "warnings": [],
  "checks": {
    "embedding_system": {
      "model_available": true,
      "index_available": true,
      "index_size": 150
    },
    "file_integrity": {
      "missing_files": [],
      "orphaned_files": [],
      "total_missing": 0,
      "total_orphaned": 0
    },
    "index_consistency": {
      "faiss_chunks": 150,
      "mapping_chunks": 150,
      "metadata_chunks": 150,
      "consistent": true
    }
  }
}
```

---

## 6. Conversation History

### 6.1 Get Chat History

Retrieve the conversation history for the current session.

- **Method:** `GET`
- **URL:** `/vision-chat/history`

### Response

```json
{
  "history": [
    {
      "role": "user",
      "content": "What are the main findings from the research documents?"
    },
    {
      "role": "assistant",
      "content": "Based on the research documents in your knowledge base, the main findings are..."
    }
  ]
}
```

### 6.2 Clear Chat History

Clear the conversation history (send with `clear_history: true` in chat request).

---

## 7. Error Handling

### Common Error Responses

**400 Bad Request:**
```json
{
  "error": "Session ID and Agent ID are required"
}
```

**404 Not Found:**
```json
{
  "error": "Document not found"
}
```

**500 Internal Server Error:**
```json
{
  "error": "Internal server error: detailed error message"
}
```

---

## 8. Integration Examples

### 8.1 Complete Chat Integration

```javascript
class VisionChatClient {
  constructor(baseUrl = 'http://127.0.0.1:5012') {
    this.baseUrl = baseUrl;
    this.sessionId = null;
    this.agentId = null;
  }

  async initialize(sessionId, agentId = 'research') {
    this.sessionId = sessionId;
    this.agentId = agentId;
    
    const response = await fetch(`${this.baseUrl}/vision-chat-initialize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        agent: agentId,
        session_id: sessionId
      })
    });
    
    return await response.json();
  }

  async chat(message, onChunk) {
    const response = await fetch(`${this.baseUrl}/vision-chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        agent: this.agentId,
        model: 'unsloth/Qwen3-1.7B-bnb-4bit',
        session_id: this.sessionId
      })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));
          if (data.response && onChunk) {
            onChunk(data.response);
          }
        }
      }
    }
  }

  async getKnowledgeBase() {
    const response = await fetch(
      `${this.baseUrl}/vision-chat/knowledge-base/${this.sessionId}?agent_id=${this.agentId}`
    );
    return await response.json();
  }

  async searchKnowledgeBase(query, maxResults = 5) {
    const response = await fetch(`${this.baseUrl}/vision-chat/search/${this.sessionId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        agent_id: this.agentId,
        max_results: maxResults
      })
    });
    return await response.json();
  }
}

// Usage
const client = new VisionChatClient();
await client.initialize('your-session-id-from-confluence');

// Start chatting
await client.chat('What are the main points from the documents?', (chunk) => {
  console.log(chunk); // Handle streaming response
});
```

### 8.2 Knowledge Base Status Check

```javascript
async function checkKnowledgeBaseHealth(sessionId, agentId) {
  const response = await fetch(
    `http://127.0.0.1:5012/vision-chat/health-check/${sessionId}?agent_id=${agentId}`
  );
  const health = await response.json();
  
  if (health.status === 'healthy') {
    console.log('Knowledge base is healthy');
  } else {
    console.warn('Knowledge base issues:', health.issues);
  }
  
  return health;
}
```

---

## 9. Best Practices

### 9.1 Session Management
- Always initialize the vision chat session before starting conversations
- Use consistent session IDs across your application
- Check knowledge base health periodically

### 9.2 Error Handling
- Implement proper error handling for all API calls
- Handle streaming connection failures gracefully
- Provide user feedback for long-running operations

### 9.3 Performance
- Use appropriate chunk sizes for streaming responses
- Monitor tokens per second for performance metrics
- Consider caching knowledge base information

### 9.4 Content Management
- Regular health checks to ensure data integrity
- Rebuild indexes after bulk operations
- Monitor embedding system availability

---

## 10. Troubleshooting

### Common Issues

**Issue: "Vision chat model not initialized"**
- Solution: Call `/vision-chat-initialize` before starting conversations

**Issue: "Embedding system not available"**
- Solution: Check if required libraries are installed and model is loaded

**Issue: "Knowledge base not found"**
- Solution: Verify session ID and ensure knowledge base was created properly

**Issue: Streaming connection drops**
- Solution: Implement reconnection logic and proper error handling

**Issue: Low relevance scores in search**
- Solution: Rebuild the embedding index or check query phrasing 