# Agent Module Documentation

## Overview

The agent module provides intelligent conversation capabilities with specialized agents for different domains. It handles agent initialization, chat interactions, streaming responses, and comprehensive response management for user applications.

## Files

### `agent.py`
Base agent class that defines the interface for all agents.

**Key Features:**
- Agent type definitions
- Base agent interface
- Agent configuration management

**Usage:**
```python
from blackbird_sdk.agent.agent import Agent

# Create custom agent
class CustomAgent(Agent):
    def process_message(self, message):
        return f"Processed: {message}"
```

### `agent_manager.py`
Manages agent lifecycle, selection, and configuration.

**Key Features:**
- Agent initialization and cleanup
- Agent type selection
- Agent configuration management
- Agent state management

**Usage:**
```python
from blackbird_sdk.agent.agent_manager import AgentManager

manager = AgentManager(http_client)
agent = manager.initialize_agent("finance")
```

### `chat_service.py`
Handles chat interactions and message processing with enhanced response extraction.

**Key Features:**
- Message sending and receiving
- Clean response text extraction
- Streaming response support
- File upload integration
- Error handling

**Usage:**
```python
from blackbird_sdk.agent.chat_service import ChatService

chat_service = ChatService(http_client, event_source_manager)

# Send regular message
response = chat_service.send_message("Hello, how are you?", options={'agent': 'finance'})

# Send streaming message
def on_chunk(chunk_text):
    print(chunk_text, end='', flush=True)

def on_complete(full_response):
    print(f"\nComplete response: {full_response}")

stream_id = chat_service.send_message(
    "Tell me about AI",
    streaming=True,
    on_chunk=on_chunk,
    on_complete=on_complete
)
```

### `chat_service_streaming.py`
Advanced streaming capabilities for real-time responses.

**Key Features:**
- Real-time response streaming
- Event-driven architecture
- Stream management (pause, resume, stop)
- Performance optimization

**Usage:**
```python
from blackbird_sdk.agent.chat_service_streaming import ChatServiceStreaming

streaming_service = ChatServiceStreaming(http_client, event_source_manager)

# Start streaming
stream_id = streaming_service.start_stream("Analyze this data")

# Pause stream
streaming_service.pause_stream(stream_id)

# Resume stream
streaming_service.resume_stream(stream_id)

# Stop stream
streaming_service.stop_stream(stream_id)
```

### `manager.py`
High-level agent management interface.

**Key Features:**
- Simplified agent operations
- Agent factory pattern
- Configuration management
- Error handling

**Usage:**
```python
from blackbird_sdk.agent.manager import AgentManager

manager = AgentManager(http_client)

# Get available agents
agents = manager.get_available_agents()

# Initialize specific agent
agent = manager.create_agent("finance", model="unsloth/Qwen3-1.7B-bnb-4bit")
```

## Response Management and User Access

### Basic Response Access

```python
# Initialize SDK
sdk = BlackbirdSDK()
sdk.initialize_agent("finance")

# Send message and get string response
response = sdk.send_message("What is the current market trend?")
print(f"Response: {response}")

# Store response for later use
my_responses = []
my_responses.append(response)

# Get full response with metadata
full_response = sdk.send_message("Analyze this data", return_full_response=True)
print(f"Agent: {full_response['agent']}")
print(f"Model: {full_response['model']}")
print(f"Response: {full_response['response']}")
```

### Streaming Response Access

```python
# Method 1: Direct streaming with callbacks
def handle_chunk(chunk_text):
    print(f"Received: {chunk_text}", end='', flush=True)

def handle_complete(full_response):
    print(f"\nComplete response: {full_response}")

sdk.send_message("Analyze this data", streaming=True, 
                on_chunk=handle_chunk, 
                on_complete=handle_complete)

# Method 2: Using StreamingResponse object
stream = sdk.stream_message("Tell me about AI trends")
stream.on_chunk(lambda chunk: print(chunk, end=''))
stream.on_complete(lambda response: print(f"\nFinal: {response}"))
stream.start()

# Wait for completion
full_response = stream.wait_for_completion()
```

### Response Storage and Management

```python
# Get conversation history
history = sdk.get_response_history(limit=5)
for item in history:
    print(f"User: {item['content']}")

# Search previous responses
results = sdk.search_responses("market analysis")
for result in results:
    print(f"Found: {result['response'][:100]}...")

# Export chat history
export_file = sdk.export_chat_history(format='txt')
print(f"Chat history exported to: {export_file}")

# Clear chat history
sdk.clear_chat_history()
```

### Interactive Chat Session

```python
# Start interactive session
sdk.chat_interactive()
# This will start a terminal-based chat interface
# Type 'quit' to exit, 'clear' to clear history
```

### Asynchronous Message Handling

```python
# Send message asynchronously
def callback(response):
    print(f"Async response: {response}")

thread = sdk.send_message_async("Hello", callback)
# Continue with other work while message processes
```

## Supported Agent Types

### 1. General Agent
- **Purpose**: General-purpose conversations
- **Capabilities**: Basic chat, information retrieval
- **Use Cases**: General questions, casual conversation

### 2. Finance Agent
- **Purpose**: Financial analysis and data processing
- **Capabilities**: 
  - Financial document analysis
  - Spreadsheet processing
  - Financial calculations
  - Market analysis
- **Use Cases**: Financial reports, investment analysis, accounting

### 3. Legal Agent
- **Purpose**: Legal document analysis
- **Capabilities**:
  - Legal document parsing
  - Contract analysis
  - Legal research
  - Compliance checking
- **Use Cases**: Contract review, legal research, compliance

### 4. Tech Agent
- **Purpose**: Code analysis and technical support
- **Capabilities**:
  - Code review and analysis
  - Debugging assistance
  - Technical documentation
  - API integration help
- **Use Cases**: Code review, debugging, technical support

### 5. Meetings Agent
- **Purpose**: Meeting transcription and analysis
- **Capabilities**:
  - Audio transcription
  - Meeting summarization
  - Action item extraction
  - Participant analysis
- **Use Cases**: Meeting notes, action items, meeting analysis

### 6. Research Agent
- **Purpose**: Research and web search capabilities
- **Capabilities**:
  - Web search integration
  - Research summarization
  - Source verification
  - Information synthesis
- **Use Cases**: Research projects, information gathering

### 7. Image-generator Agent
- **Purpose**: Image generation and manipulation
- **Capabilities**:
  - Text-to-image generation
  - Image editing
  - Style transfer
  - Image analysis
- **Use Cases**: Content creation, design, image generation

## Agent Configuration

### Model Selection
```python
# Use default model
sdk.initialize_agent("finance")

# Use specific model
sdk.initialize_agent("finance", model="unsloth/Qwen3-1.7B-bnb-4bit")
```

### Agent Options
```python
options = {
    "temperature": 0.7,
    "max_tokens": 1000,
    "stream": True
}

sdk.initialize_agent("finance", options=options)
```

## Streaming Support

### Basic Streaming
```python
def on_chunk(chunk_text):
    print(chunk_text, end='', flush=True)

def on_complete():
    print("\nStream completed")

def on_error(error):
    print(f"Error: {error}")

sdk.send_message(
    "Analyze this financial data",
    streaming=True,
    on_chunk=on_chunk,
    on_complete=on_complete,
    on_error=on_error
)
```

### Advanced Streaming with Real-time Access
```python
# Create streaming response object
stream = sdk.stream_message("Start comprehensive analysis")

# Add multiple chunk handlers
stream.on_chunk(lambda chunk: print(chunk, end=''))
stream.on_chunk(lambda chunk: save_to_file(chunk))

# Add completion handler
stream.on_complete(lambda response: process_final_response(response))

# Start streaming
stream.start()

# Get current response while streaming
current = stream.get_current_response()
print(f"So far: {current}")

# Wait for completion with timeout
try:
    final_response = stream.wait_for_completion(timeout=60)
    print(f"Final response: {final_response}")
except TimeoutError:
    print("Stream timed out")
```

### Stream Management
```python
# Start stream
stream_id = sdk.send_streaming_message("Start analysis")

# Check stream status
status = sdk.get_stream_health()
print(f"Active streams: {status['active_streams']}")

# Stop specific stream
sdk.stop_stream(stream_id)

# Stop all streams
sdk.stop_all_streams()
```

## Response Processing and Storage

### Response Manager Integration
```python
# The SDK automatically stores all responses
# Access them through the response manager

# Get recent responses
recent = sdk.response_manager.get_responses(limit=10)
for response in recent:
    print(f"Agent: {response.agent}")
    print(f"Message: {response.message}")
    print(f"Response: {response.response}")
    print(f"Timestamp: {response.timestamp}")

# Search responses
results = sdk.response_manager.search_responses("financial analysis")
for result in results:
    print(f"Found match: {result.response[:100]}...")

# Get conversation history for chat applications
history = sdk.response_manager.get_conversation_history(limit=20)
# Returns format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
```

### Custom Response Processing
```python
# Add custom response processing
def process_response(response_text, agent, model):
    # Custom processing logic
    processed = response_text.upper()  # Example transformation
    
    # Store in custom format
    with open("custom_responses.txt", "a") as f:
        f.write(f"[{agent}] {processed}\n")
    
    return processed

# Use with SDK
response = sdk.send_message("Hello")
processed = process_response(response, sdk.current_agent, sdk.current_model)
```

## Error Handling

### Common Errors
1. **ValidationError**: Invalid input or missing agent
2. **StreamingResponseError**: Streaming not available
3. **NetworkError**: Connection issues
4. **TimeoutError**: Request timeout

### Enhanced Error Handling
```python
from blackbird_sdk.utils.errors import ValidationError, StreamingResponseError

try:
    sdk.initialize_agent("finance")
    response = sdk.send_message("Hello")
    
    # Process response
    if isinstance(response, dict):
        text = response.get('response', str(response))
    else:
        text = str(response)
    
    print(f"Response: {text}")
    
except ValidationError as e:
    print(f"Validation error: {e.message}")
    if e.field_name:
        print(f"Field: {e.field_name}")
        
except StreamingResponseError as e:
    print(f"Streaming error: {e.message}")
    if e.fallback_available:
        print("Trying non-streaming mode...")
        response = sdk.send_message("Hello", streaming=False)
        
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Optimization

### Response Caching
```python
# Responses are automatically cached by the response manager
# Access cached responses
cache_stats = sdk.response_manager.get_cache_stats()
print(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")

# Clear cache if needed
sdk.response_manager.clear_cache()
```

### Memory Management
```python
# Monitor memory usage
memory_stats = sdk.response_manager.get_vector_stats()
print(f"Documents indexed: {memory_stats['total_documents']}")
print(f"Index size: {memory_stats['index_size_mb']:.2f} MB")

# Cleanup when done
sdk.cleanup()
```

### Batch Processing
```python
# Process multiple messages efficiently
messages = ["Hello", "How are you?", "What's the weather?"]
responses = []

for message in messages:
    response = sdk.send_message(message)
    responses.append(response)

# Or use async processing
import asyncio

async def process_messages(messages):
    tasks = []
    for message in messages:
        task = asyncio.create_task(
            asyncio.to_thread(sdk.send_message, message)
        )
        tasks.append(task)
    
    return await asyncio.gather(*tasks)

# responses = asyncio.run(process_messages(messages))
```

## Testing

### Unit Tests
```python
import pytest
from blackbird_sdk import BlackbirdSDK

def test_agent_initialization():
    sdk = BlackbirdSDK(development_mode=True)
    response = sdk.initialize_agent("finance")
    assert response is not None
    assert sdk.current_agent == "finance"

def test_message_sending():
    sdk = BlackbirdSDK(development_mode=True)
    sdk.initialize_agent("finance")
    response = sdk.send_message("Hello")
    assert response is not None
    assert isinstance(response, str)

def test_response_storage():
    sdk = BlackbirdSDK(development_mode=True)
    sdk.initialize_agent("finance")
    
    # Send message
    response = sdk.send_message("Test message")
    
    # Check if stored
    history = sdk.get_response_history(limit=1)
    assert len(history) > 0
    assert "Test message" in str(history)
```

### Integration Tests
```python
def test_streaming_functionality():
    sdk = BlackbirdSDK(development_mode=True)
    sdk.initialize_agent("finance")
    
    chunks = []
    def on_chunk(chunk):
        chunks.append(chunk)
    
    def on_complete(response):
        assert len(chunks) > 0
        assert response == ''.join(chunks)
    
    sdk.send_message("Test streaming", streaming=True, 
                    on_chunk=on_chunk, on_complete=on_complete)

def test_response_management():
    sdk = BlackbirdSDK(development_mode=True)
    sdk.initialize_agent("finance")
    
    # Send multiple messages
    sdk.send_message("First message")
    sdk.send_message("Second message")
    
    # Test search
    results = sdk.search_responses("First")
    assert len(results) > 0
    
    # Test export
    export_file = sdk.export_chat_history()
    assert os.path.exists(export_file)
```

## Best Practices

1. **Agent Selection**: Choose the most appropriate agent for your task
2. **Response Handling**: Always extract clean text from responses
3. **Streaming**: Use streaming for long responses and real-time applications
4. **Error Handling**: Implement comprehensive error handling
5. **Resource Management**: Clean up resources when done
6. **Response Storage**: Leverage built-in response management for conversation history
7. **Memory Management**: Monitor and manage memory usage for large applications

## Troubleshooting

### Common Issues

1. **Agent not responding**
   - Check network connectivity
   - Verify agent type is supported
   - Check model availability
   - Ensure backend is running

2. **Streaming not working**
   - Enable streaming feature flag
   - Check EventSourceManager initialization
   - Verify streaming support in backend

3. **Response format issues**
   - Use response extraction methods
   - Check for dict vs string responses
   - Implement proper type checking

4. **Memory issues**
   - Monitor response storage
   - Clear cache periodically
   - Use cleanup methods

### Debug Mode
```python
# Enable debug logging
sdk = BlackbirdSDK(development_mode=True)

# Check backend status
status = sdk.get_backend_status()
print(f"Backend running: {status['is_running']}")

# Monitor response storage
stats = sdk.response_manager.get_cache_stats()
print(f"Responses stored: {stats['total_entries']}")
```

## API Reference

### BlackbirdSDK Main Methods
- `send_message(message, streaming=False, return_full_response=False, **kwargs)`: Send message with options
- `stream_message(message)`: Create streaming response object
- `send_message_async(message, callback)`: Send message asynchronously
- `chat_interactive()`: Start interactive chat session

### Response Management
- `get_response_history(limit=10)`: Get conversation history
- `search_responses(query)`: Search through responses
- `export_chat_history(format='json', output_path=None)`: Export chat history
- `clear_chat_history()`: Clear all stored responses

### AgentManager
- `initialize_agent(agent_type, model_name=None)`: Initialize agent
- `get_available_agents()`: Get list of available agents
- `get_agent_capabilities(agent_type)`: Get agent capabilities

### ChatService
- `send_message(message, options=None, streaming=False)`: Send message
- `_extract_response_text(response)`: Extract clean text from response
- `stop_stream(stream_id)`: Stop streaming
- `get_stream_status(stream_id)`: Get stream status

### StreamingResponse
- `on_chunk(callback)`: Add chunk callback
- `on_complete(callback)`: Add completion callback
- `start()`: Start streaming
- `wait_for_completion(timeout=30)`: Wait for completion
- `get_current_response()`: Get current accumulated response

### ResponseManager
- `add_response(message, response, agent, model)`: Add new response
- `get_responses(limit, agent)`: Get filtered responses
- `get_conversation_history(limit)`: Get chat format history
- `search_responses(query)`: Search response content
- `export_responses(format, output_path)`: Export responses
- `clear_responses()`: Clear all responses
