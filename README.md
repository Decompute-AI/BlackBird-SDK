# Blackbird Open Source SDK

A powerful, open-source AI SDK for local inference, RAG, file processing. Built for privacy, performance, and ease of use.

## 🚀 Features

- **Local AI Inference**: Run AI models locally without cloud dependencies
- **RAG (Retrieval-Augmented Generation)**: Process documents and files for context-aware responses
- **File Processing**: Upload and process various document formats
- **Streaming Responses**: Real-time streaming chat responses
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **MIT Licensing**

## 📋 Requirements

### System Requirements
- **Windows**: Windows 10/11 (64-bit)
- **macOS**: macOS 10.15+ (Intel/Apple Silicon)
- **Linux**: Ubuntu 18.04+ or equivalent
- **Python**: 3.12
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: 10GB+ free space for models

### Hardware Requirements
- **CPU**: Multi-core processor (Intel/AMD/Apple Silicon)
- **GPU**: Optional but recommended for better performance
  - **NVIDIA**: CUDA-compatible GPU (4GB+ VRAM)
  - **Apple Silicon**: Built-in Neural Engine

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/blackbird-open-source-sdk.git
cd blackbird-open-source-sdk
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

#### Windows
```bash
pip install -r requirements_windows.txt
```

#### macOS
```bash
pip install -r requirements_mac.txt
```

#### Linux
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "from open_source_sdk import BlackbirdSDK; print('✅ SDK installed successfully!')"
```

## 🚀 Quick Start

### Basic Chat Example
```python
from open_source_sdk import BlackbirdSDK

# Initialize the SDK
sdk = BlackbirdSDK()

# Initialize an agent
sdk.initialize_agent("general")

# Send a message
response = sdk.send_message("Hello! What can you help me with?")
print(response)
```

### Streaming Chat Example
```python
from open_source_sdk import BlackbirdSDK

sdk = BlackbirdSDK()

# Initialize agent
sdk.initialize_agent("general")

# Send streaming message
response = sdk.send_message("Tell me a story about AI.", streaming=True)
print(response)
```

## 📚 Examples

### Basic Examples
- `examples/chat_response.py` - Simple chat interaction
- `examples/test_sdk_minimal.py` - Minimal SDK test
- `examples/chat_streaming_ui.py` - GUI chat application
## 🎯 Usage Guide

### 1. Backend Management

The SDK automatically manages the backend server. For manual control:

```python
# Start backend in keepalive mode (recommended for development)
sdk = BlackbirdSDK(runasync=True)

# Normal mode (backend starts/stops with SDK)
sdk = BlackbirdSDK()
```

### 2. Agent Types

Available agent types:
- `general` - General purpose assistant
- `coding` - Programming and code analysis
- `research` - Research from academic journals
- `finance` - Data analysis and research
### 3. File Processing

```python
# Upload and process files
result = sdk.upload_file("document.pdf", agent_type="general")

# Initialize agent with files
result = sdk.initialize_agent_with_files("general", ["file1.pdf", "file2.docx"])
```

### 4. Streaming Responses

```python
# Real-time streaming
def on_chunk(chunk):
    print(chunk, end='', flush=True)

def on_complete(response):
    print(f"\nComplete response: {response}")

sdk.send_streaming_message(
    "Explain quantum computing",
    on_chunk=on_chunk,
    on_complete=on_complete
)
```

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set Hugging Face token for model downloads
export HF_TOKEN=your_token_here

# Optional: Set custom model cache directory
export BLACKBIRD_CACHE_DIR=/path/to/cache
```

### SDK Configuration
```python
sdk = BlackbirdSDK(
    log_level='INFO',           # Logging level
    development_mode=False,     # Show internal logs
    user_logging=True,          # User-friendly messages
    offline_mode=False          # Offline-only mode
)
```

## 🏗️ Architecture

```
open_source_sdk/
├── __init__.py                 # Main SDK entry point
├── agent/                      # Agent management
├── oss_utils/                  # Core utilities
├── oss_session/                # Session and memory management
├── oss_model/                  # Model management
├── oss_acceleration/           # Hardware acceleration
├── oss_data_pipeline/          # File processing
├── creation/                   # Agent creation tools
├── integrations/               # Third-party integrations
├── server/                     # Backend server management
├── examples/                   # Usage examples
└── backends/                   # Backend implementations
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Ensure you're in the correct directory
cd open_source_sdk

# Add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 2. Backend Connection Issues
```bash
# Check if port 5012 is available
netstat -an | grep 5012

# Force cleanup if needed
python -c "from open_source_sdk.server.backend_manager import BackendManager; BackendManager.get_instance().force_cleanup()"
```

#### 3. Model Download Issues
```bash
# Clear model cache
rm -rf ~/.cache/huggingface/

# Set HF token
export HF_TOKEN=your_token_here
```

#### 4. Memory Issues
```python
# Use smaller models
sdk.initialize_agent("general", model_name="unsloth/Qwen3-1.7B-bnb-4bit")

# Enable cleanup
sdk.cleanup()
```

### Performance Optimization

#### Windows
- Enable CUDA acceleration if available
- Use WSL2 for better performance
- Allocate sufficient RAM to WSL2

#### macOS
- Use Apple Silicon models for better performance
- Enable MPS acceleration
- Monitor Activity Monitor for memory usage

#### Linux
- Install CUDA drivers if using NVIDIA GPU
- Use system package manager for dependencies
- Monitor system resources

## 📖 API Reference

### Core Classes

#### BlackbirdSDK
Main SDK class for all operations.

```python
sdk = BlackbirdSDK(
    license_server_url=None,
    log_level='INFO',
    structured_logging=True,
    feature_config=None,
    development_mode=False,
    user_logging=True,
    offline_mode=False,
    web_search_backend=None,
    skip_licensing=True,
    runasync=False
)
```

#### Key Methods
- `initialize_agent(agent_type, model_name=None)` - Initialize an agent
- `send_message(message, streaming=False, **kwargs)` - Send a message
- `upload_file(file_path, agent_type=None)` - Upload a file
- `cleanup()` - Clean up resources

### Agent Management

#### Available Agents
- `general` - General purpose
- `coding` - Programming assistant
- `writing` - Content creation
- `analysis` - Data analysis
- `creative` - Creative tasks



## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone repository
git clone https://github.com/your-username/blackbird-open-source-sdk.git
cd blackbird-open-source-sdk

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Run tests
python -m pytest tests/

# Run linting
flake8 open_source_sdk/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support


- **Examples**: [examples/](examples/)

## 🙏 Acknowledgments

- Hugging Face for model hosting
- Unsloth for model optimization
- The open source AI community

## 📈 Roadmap

- [ ] Additional model support
- [ ] Enhanced file processing
- [ ] Performance optimizations
- [ ] Mobile support

---

**Made with ❤️ by the Decompute Team** 
