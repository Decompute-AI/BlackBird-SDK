#!/usr/bin/env python3
"""
Environment Setup Script for Web Research Pipeline
Helps users configure their .env file with API keys and settings
"""

import os
import sys
from pathlib import Path
import shutil

def create_env_file():
    """Create .env file from template"""
    
    # Get the current directory
    current_dir = Path(__file__).parent
    env_example = current_dir / "env_example.txt"
    env_file = current_dir / ".env"
    
    if env_file.exists():
        print(f"⚠️  .env file already exists at: {env_file}")
        response = input("Do you want to overwrite it? (y/N): ").lower()
        if response != 'y':
            print("Setup cancelled.")
            return False
    
    if not env_example.exists():
        print(f"❌ Template file not found: {env_example}")
        return False
    
    try:
        # Copy the template
        shutil.copy2(env_example, env_file)
        print(f"✅ Created .env file at: {env_file}")
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def get_api_key_input(service_name, description, required=False):
    """Get API key input from user"""
    print(f"\n{'='*60}")
    print(f"🔑 {service_name.upper()} API KEY")
    print(f"   {description}")
    
    if required:
        print("   ⚠️  REQUIRED for web search functionality")
    else:
        print("   ℹ️  Optional - can be added later")
    
    api_key = input(f"   Enter your {service_name} API key (or press Enter to skip): ").strip()
    
    if required and not api_key:
        print(f"   ❌ {service_name} API key is required!")
        return get_api_key_input(service_name, description, required)
    
    return api_key

def update_env_file(env_file_path, updates):
    """Update .env file with user input"""
    try:
        # Read the current .env file
        with open(env_file_path, 'r') as f:
            content = f.read()
        
        # Update the content with user input
        for key, value in updates.items():
            if value:  # Only update if value is provided
                # Replace the placeholder with actual value
                placeholder = f"{key}=your_{key.lower()}_key_here"
                replacement = f"{key}={value}"
                content = content.replace(placeholder, replacement)
        
        # Write back to file
        with open(env_file_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated .env file with your API keys")
        return True
        
    except Exception as e:
        print(f"❌ Error updating .env file: {e}")
        return False

def validate_env_file(env_file_path):
    """Validate the .env file"""
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file_path)
        
        # Check required keys
        required_keys = {
            'PERPLEXITY_API_KEY': 'Required for web search functionality'
        }
        
        missing_keys = []
        for key, description in required_keys.items():
            if not os.getenv(key):
                missing_keys.append(f"{key}: {description}")
        
        if missing_keys:
            print(f"\n❌ Missing required configuration:")
            for missing in missing_keys:
                print(f"   - {missing}")
            return False
        else:
            print(f"\n✅ All required configuration is present!")
            return True
            
    except ImportError:
        print("⚠️  python-dotenv not installed. Please install it with: pip install python-dotenv")
        return False
    except Exception as e:
        print(f"❌ Error validating .env file: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print(f"\n{'='*60}")
    print("🎉 Environment Setup Complete!")
    print(f"{'='*60}")
    
    print("\n📋 Next Steps:")
    print("1. Test your configuration:")
    print("   python test_web_search.py")
    
    print("\n2. Run the example usage:")
    print("   python example_usage.py")
    
    print("\n3. Integrate with Vision Chat:")
    print("   The web search is automatically integrated when PERPLEXITY_API_KEY is set")
    
    print("\n4. Monitor your API usage:")
    print("   - Perplexity: Check your dashboard at https://perplexity.ai")
    print("   - Keep track of rate limits and quotas")
    
    print("\n🔒 Security Notes:")
    print("- Never commit your .env file to version control")
    print("- Keep your API keys secure")
    print("- Rotate your keys regularly")
    print("- Monitor for unusual API usage")
    
    print("\n📚 Documentation:")
    print("- README_WEB_SEARCH.md - Complete documentation")
    print("- IMPLEMENTATION_SUMMARY.md - Technical details")
    print("- example_usage.py - Usage examples")

def main():
    """Main setup function"""
    print("🚀 Web Research Pipeline Environment Setup")
    print("=" * 60)
    
    # Step 1: Create .env file
    print("\n📝 Step 1: Creating .env file...")
    if not create_env_file():
        return
    
    # Step 2: Get API keys from user
    print("\n🔑 Step 2: Configure API Keys...")
    
    api_keys = {}
    
    # Required API keys
    api_keys['PERPLEXITY_API_KEY'] = get_api_key_input(
        'perplexity',
        'Get your API key from: https://perplexity.ai',
        required=True
    )
    
    # Optional API keys
    api_keys['BRAVE_API_KEY'] = get_api_key_input(
        'brave',
        'Alternative search provider - Get from: https://brave.com/search/api/'
    )
    
    api_keys['NEWSAPI_KEY'] = get_api_key_input(
        'newsapi',
        'News-specific searches - Get from: https://newsapi.org'
    )
    
    api_keys['SEMANTIC_SCHOLAR_KEY'] = get_api_key_input(
        'semantic_scholar',
        'Academic research - Get from: https://www.semanticscholar.org/product/api'
    )
    
    api_keys['OPENAI_API_KEY'] = get_api_key_input(
        'openai',
        'Advanced query generation - Get from: https://platform.openai.com'
    )
    
    # Get email for PubMed
    print(f"\n{'='*60}")
    print("📧 PUBMED EMAIL")
    print("   Required for academic searches")
    print("   Use your institutional or personal email")
    
    pubmed_email = input("   Enter your email for PubMed API: ").strip()
    if pubmed_email:
        api_keys['PUBMED_EMAIL'] = pubmed_email
    
    # Step 3: Update .env file
    print("\n💾 Step 3: Updating .env file...")
    env_file_path = Path(__file__).parent / ".env"
    
    if update_env_file(env_file_path, api_keys):
        print("✅ Configuration saved successfully!")
    else:
        print("❌ Failed to save configuration")
        return
    
    # Step 4: Validate configuration
    print("\n✅ Step 4: Validating configuration...")
    if validate_env_file(env_file_path):
        print("✅ Configuration is valid!")
    else:
        print("⚠️  Configuration has issues. Please check the .env file manually.")
    
    # Step 5: Print next steps
    print_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user.")
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        print("Please check the error and try again.") 