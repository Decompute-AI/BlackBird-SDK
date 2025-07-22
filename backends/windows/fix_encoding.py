#!/usr/bin/env python
# fix_encoding.py - UTF-8 encoding fix for application
"""
This script fixes the Unicode encoding issue with emoji and other special characters
when printing to the console or redirecting output on Windows.

It configures the system to use UTF-8 for I/O operations by:
1. Setting the PYTHONIOENCODING environment variable
2. Reconfiguring sys.stdout and sys.stderr
3. Setting the console code page to UTF-8 on Windows

Run this script before starting your application, or import it at the beginning of your main script.
"""

import sys
import os
import locale

def setup_utf8_io():
    """Configure the Python environment to use UTF-8 encoding for all I/O operations."""
    
    # 1. Set PYTHONIOENCODING environment variable
    os.environ["PYTHONIOENCODING"] = "utf-8"
    
    # 2. Reconfigure stdout and stderr
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
    
    # 3. Set the console code page to UTF-8 on Windows
    if sys.platform == 'win32':
        try:
            import subprocess
            subprocess.run(['chcp', '65001'], check=False, shell=True)
            print("Console code page set to UTF-8 (65001)")
        except Exception as e:
            print(f"Warning: Could not set console code page: {e}")
    
    # 4. Set the locale to use UTF-8
    try:
        locale.setlocale(locale.LC_ALL, '.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, '')  # Use default locale
        except Exception as e:
            print(f"Warning: Could not set locale: {e}")
    
    # Print current encoding settings
    print(f"sys.stdout encoding: {sys.stdout.encoding}")
    print(f"sys.stderr encoding: {sys.stderr.encoding}")
    print(f"Default encoding: {sys.getdefaultencoding()}")
    print(f"Filesystem encoding: {sys.getfilesystemencoding()}")
    
    # Test with an emoji
    emoji_test = "✅ UTF-8 encoding test successful! 🚀"
    
    # Try to safely print the test line
    try:
        print(emoji_test)
        print("Emoji display test passed")
    except UnicodeEncodeError:
        # If it still fails, we'll print a fallback message
        print("Note: Your console may not support emoji display, but UTF-8 encoding is configured")

if __name__ == "__main__":
    setup_utf8_io()
    print("UTF-8 encoding configured successfully. You can now run your application.") 