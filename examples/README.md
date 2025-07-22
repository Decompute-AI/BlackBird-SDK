# Blackbird Open Source SDK - Examples

This directory contains comprehensive examples demonstrating how to use the Blackbird Open Source SDK.

## 📚 Example Categories

### 🚀 Basic Examples
Start here if you're new to the SDK.

| Example | Description | Difficulty |
|---------|-------------|------------|
| `chat_response.py` | Simple chat interaction | ⭐ |
| `chat_streaming_ui.py` | GUI chat application with streaming | ⭐⭐ |



### 📖 Documentation
| File | Description |
|------|-------------|
| `chat_demo.md` | Comprehensive chat demo guide |

## 🚀 Quick Start Examples

### 1. Simple Chat
```bash
python examples/chat_response.py
```
**What it does:** Demonstrates basic chat functionality with user-friendly messages.

### 2. GUI Chat Application
```bash
python examples/chat_streaming_ui.py
```
**What it does:** Opens a graphical chat interface with real-time streaming.

## 📖 Detailed Documentation

### Chat Demo Guide
Read `chat_demo.md` for comprehensive information about:
- Streaming chat implementation
- Backend management
- Response handling
- UI integration
- Troubleshooting

## 🎯 Example Outputs

### Expected Output from `test_sdk_minimal.py`
```
✅ Successfully imported BlackbirdSDK from open_source_sdk
[timestamp] ✅ Checking services...
[timestamp] ✅ Starting services...
✅ SDK initialized successfully
✅ Agent initialized successfully
✅ Message sent successfully
Chat response: Hello! I'm here to help you with various tasks...
```

### Expected Output from `chat_response.py`
```
✅ Successfully imported BlackbirdSDK from open_source_sdk
[timestamp] ✅ Checking services...
[timestamp] ✅ Starting services...
✅ SDK initialized successfully
✅ Agent initialized successfully
✅ Message sent successfully
Chat response: Hello! What is the weather today? I don't have access to real-time weather data...
```

## 🔧 Customizing Examples

### Modifying Agent Types
```python
# Change agent type
sdk.initialize_agent("coding")  # Instead of "general"
```

### Using Different Models
```python
# Use specific model
sdk.initialize_agent("general", model_name="unsloth/Qwen3-1.7B-bnb-4bit")
```

### Custom Agent Configuration
```python
# Create custom agent with specific personality
agent = (sdk.create_custom_agent("My Agent", "Custom description")
    .personality("creative")
    .system_prompt("You are a creative writing assistant.")
    .temperature(0.8)
    .build(sdk_instance=sdk))
```

## 🐛 Troubleshooting Examples

### Common Issues

#### 1. Import Errors
```bash
# Solution: Run from correct directory
cd open_source_sdk
python examples/chat_response.py
```

#### 2. Backend Issues
```bash
# Solution: Force cleanup
python -c "from open_source_sdk.server.backend_manager import BackendManager; BackendManager.get_instance().force_cleanup()"
```

#### 3. Model Download Issues
```bash
# Solution: Use smaller model
# Modify examples to use: model_name="unsloth/Qwen3-1.7B-bnb-4bit"
```

## 📝 Creating Your Own Examples

### Basic Template
```python
#!/usr/bin/env python3
"""
Your Example Name
Description of what this example demonstrates
"""

import sys
import os

# Add the open_source_sdk directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from open_source_sdk import BlackbirdSDK

def main():
    """Main function demonstrating the feature."""
    try:
        # Initialize SDK
        sdk = BlackbirdSDK()
        print("✅ SDK initialized successfully")
        
        # Your example code here
        sdk.initialize_agent("general")
        response = sdk.send_message("Your test message")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
```

### Best Practices
1. **Error Handling**: Always include try-catch blocks
2. **User Feedback**: Use clear success/error messages
3. **Documentation**: Include docstrings and comments
4. **Path Management**: Handle import paths correctly
5. **Cleanup**: Clean up resources when done


**Happy experimenting with Blackbird Open Source SDK! 🚀** 
