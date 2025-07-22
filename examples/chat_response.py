# chat_response.py - Open Source SDK Test

import sys
import os

# Add the open_source_sdk directory to the Python path (same as working test script)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'open_source_sdk'))

try:
    from open_source_sdk import BlackbirdSDK
    print("✅ Successfully imported BlackbirdSDK from open_source_sdk")
    
    # Test initialization
    sdk = BlackbirdSDK()
    print("✅ SDK initialized successfully")
    
    # Test agent initialization
    sdk.initialize_agent("general")
    print("✅ Agent initialized successfully")
    
    # Test message sending
    response = sdk.send_message("Hello, what is the weather today?")
    print("✅ Message sent successfully")
    print("Chat response:", response)
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}") 