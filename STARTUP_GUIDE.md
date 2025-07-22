# Blackbird Open Source SDK - Startup Guide

This guide will help you get the Blackbird Open Source SDK up and running on your system.

## 🎯 Quick Start (5 minutes)

### Step 1: Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/blackbird-open-source-sdk.git
cd blackbird-open-source-sdk

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### Step : Test Installation
```bash
python examples/test_sdk_minimal.py
```

### Step 4: Run Your First Chat
```bash
python examples/chat_response.py
```

## 🖥️ Platform-Specific Setup

### Windows Setup

#### Prerequisites
- Windows 10/11 (64-bit)
- Python 3.8-3.11
- 8GB+ RAM
- Visual Studio Build Tools (for some packages)

#### Installation Steps
```bash
# 1. Install Python (if not already installed)
# Download from https://www.python.org/downloads/

# 2. Install Visual Studio Build Tools (if needed)
# Download from https://visualstudio.microsoft.com/visual-cpp-build-tools/

# 3. Clone and setup
git clone https://github.com/your-username/blackbird-open-source-sdk.git
cd blackbird-open-source-sdk

# 4. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 5. Install dependencies
pip install -r requirements_windows.txt

# 6. Test installation
python examples/test_sdk_minimal.py
```

#### Troubleshooting Windows Issues
```bash
# If you get build errors:
pip install --upgrade pip setuptools wheel

# If CUDA issues occur:
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# If port 5012 is in use:
netstat -ano | findstr :5012
taskkill /PID <PID> /F
```

### macOS Setup

#### Prerequisites
- macOS 10.15+ (Intel or Apple Silicon)
- Python 3.8-3.11
- 8GB+ RAM
- Xcode Command Line Tools

#### Installation Steps
```bash
# 1. Install Xcode Command Line Tools
xcode-select --install

# 2. Install Python (if not already installed)
# Download from https://www.python.org/downloads/

# 3. Clone and setup
git clone https://github.com/your-username/blackbird-open-source-sdk.git
cd blackbird-open-source-sdk

# 4. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 5. Install dependencies
pip install -r requirements_mac.txt

# 6. Test installation
python examples/test_sdk_minimal.py
```

#### Apple Silicon Optimization
```bash
# For better performance on Apple Silicon:
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Use Apple Silicon optimized models
sdk.initialize_agent("general", model_name="mlx-community/Qwen3-4B-4bit")
```

### Linux Setup

#### Prerequisites
- Ubuntu 18.04+ or equivalent
- Python 3.8-3.11
- 8GB+ RAM
- Build essentials

#### Installation Steps
```bash
# 1. Install system dependencies
sudo apt update
sudo apt install python3 python3-pip python3-venv build-essential

# 2. Clone and setup
git clone https://github.com/your-username/blackbird-open-source-sdk.git
cd blackbird-open-source-sdk

# 3. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Test installation
python examples/test_sdk_minimal.py
```

#### NVIDIA GPU Setup (Optional)
```bash
# Install CUDA drivers
sudo apt install nvidia-cuda-toolkit

# Install PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 Running Examples

### Basic Examples

#### 1. Simple Chat
```bash
python examples/chat_response.py
```
**Expected Output:**
```
✅ Successfully imported BlackbirdSDK from open_source_sdk
[timestamp] ✅ Checking services...
[timestamp] ✅ Starting services...
✅ SDK initialized successfully
✅ Agent initialized successfully
✅ Message sent successfully
Chat response: Hello! I'm here to help you with various tasks...
```

#### 2. Streaming Chat
```bash
python examples/chat_streaming_ui.py
```
This opens a GUI window for interactive chat.




## 🔧 Configuration Options

### Environment Variables
```bash
# Set Hugging Face token (optional, for private models)
export HF_TOKEN=your_token_here

# Set custom cache directory
export BLACKBIRD_CACHE_DIR=/path/to/cache

# Enable development mode
export BLACKBIRD_DEV_MODE=1

# Set log level
export BLACKBIRD_LOG_LEVEL=DEBUG
```

### SDK Configuration
```python
from open_source_sdk import BlackbirdSDK

# Basic configuration
sdk = BlackbirdSDK(
    log_level='INFO',
    development_mode=False,
    user_logging=True
)

# Development configuration
sdk = BlackbirdSDK(
    log_level='DEBUG',
    development_mode=True,
    user_logging=True
)

# Async backend mode (recommended for development)
sdk = BlackbirdSDK(runasync=True)
```

## 🐛 Common Issues and Solutions

### Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'open_source_sdk'

# Solution 1: Add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Solution 2: Run from correct directory
cd open_source_sdk
python examples/chat_response.py

# Solution 3: Install in development mode
pip install -e .
```

### Backend Connection Issues
```bash
# Error: Connection refused on port 5012

# Solution 1: Check if port is in use
netstat -an | grep 5012

# Solution 2: Force cleanup
python -c "from open_source_sdk.server.backend_manager import BackendManager; BackendManager.get_instance().force_cleanup()"

# Solution 3: Kill process manually
# Windows:
netstat -ano | findstr :5012
taskkill /PID <PID> /F
# macOS/Linux:
lsof -ti:5012 | xargs kill -9
```

### Model Download Issues
```bash
# Error: Failed to download model

# Solution 1: Clear cache
rm -rf ~/.cache/huggingface/

# Solution 2: Set HF token
export HF_TOKEN=your_token_here

# Solution 3: Use smaller model
sdk.initialize_agent("general", model_name="unsloth/Qwen3-1.7B-bnb-4bit")
```

### Memory Issues
```bash
# Error: CUDA out of memory or system memory issues

# Solution 1: Use smaller model
sdk.initialize_agent("general", model_name="unsloth/Qwen3-1.7B-bnb-4bit")

# Solution 2: Enable cleanup
sdk.cleanup()

# Solution 3: Reduce batch size (if applicable)
# Add to your code:
import torch
torch.cuda.empty_cache()
```

### Performance Issues
```bash
# Slow inference

# Solution 1: Enable hardware acceleration
# Windows: Install CUDA drivers
# macOS: Use Apple Silicon models
# Linux: Install CUDA drivers

# Solution 2: Use optimized models
# Windows/Linux: unsloth/Qwen3-1.7B-bnb-4bit
# macOS: mlx-community/Qwen3-4B-4bit

# Solution 3: Use async backend mode
sdk = BlackbirdSDK(runasync=True)
```

## 📊 Performance Benchmarks

### Model Performance (Approximate)
| Model | Platform | Memory | Speed | Quality |
|-------|----------|--------|-------|---------|
| Qwen3-1.7B-4bit | Windows/Linux | 4GB | Fast | Good |
| Qwen3-4B-4bit | macOS | 6GB | Medium | Better |
| Qwen3-7B-4bit | All | 8GB | Slow | Best |

### Hardware Recommendations
| Use Case | CPU | RAM | GPU | Storage |
|----------|-----|-----|-----|---------|
| Basic Chat | 4 cores | 8GB | Optional | 10GB |
| File Processing | 8 cores | 16GB | Recommended | 20GB |
| Development | 8+ cores | 16GB+ | Recommended | 50GB+ |

## 🔄 Development Workflow

### 1. Daily Development
```bash
# Start async backend (keep running)
python -c "from open_source_sdk import BlackbirdSDK; sdk = BlackbirdSDK(runasync=True)"

# In another terminal, run your code
python your_script.py
```

### 2. Testing
```bash
# Run all examples
for file in examples/*.py; do
    echo "Running $file..."
    python "$file"
done

# Run specific test
python examples/test_sdk_minimal.py
```

### 3. Debugging
```bash
# Enable debug mode
export BLACKBIRD_LOG_LEVEL=DEBUG
python your_script.py

# Or in code
sdk = BlackbirdSDK(development_mode=True, log_level='DEBUG')
```

## 📚 Next Steps

After successful installation:

1. **Read the Examples**: Explore `examples/` directory
2. **Check Documentation**: Read `README.md` and `chat_demo.md`
3. **Try Custom Agents**: Use agent creation examples
4. **Explore Features**: Test file processing, web search, etc.
5. **Join Community**: Check GitHub issues and discussions

## 🆘 Getting Help

- **Documentation**: Check `README.md` and `chat_demo.md`
- **Examples**: Run examples in `examples/` directory
- **Issues**: Search existing issues on GitHub
- **Discussions**: Ask questions in GitHub Discussions
- **Debug Mode**: Enable debug logging for detailed error information

---

**Happy coding with Blackbird Open Source SDK! 🚀** 
