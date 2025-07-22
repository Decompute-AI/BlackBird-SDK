from cryptography.fernet import Fernet
import os
import tempfile
import shutil
import uuid

def encrypt_model_directory(source_path, output_path):
    """
    Encrypt an entire model directory
    Args:
        source_path: Path to the model directory to encrypt
        output_path: Path to save the encrypted model (defaults to source_path + "_encrypted")
    Returns:
        Tuple of (output_path, encryption_key)
    """
    if output_path is None:
        output_path = source_path + "_encrypted"
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Generate encryption key
    encryption_key = Fernet.generate_key()
    cipher = Fernet(encryption_key)
    file_count = 0
    # Encrypt each file in the model directory
    for root, dirs, files in os.walk(source_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Get relative path to maintain directory structure
            rel_path = os.path.relpath(file_path, source_path)
            output_file = os.path.join(output_path, rel_path)
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            # Encrypt file contents
            with open(file_path, 'rb') as f:
                data = f.read()
            encrypted_data = cipher.encrypt(data)
            # Save encrypted file
            with open(output_file, 'wb') as f:
                f.write(encrypted_data)
            file_count += 1
    # Save the encryption key
    key_path = os.path.join(output_path, "encryption_key.bin")

    with open(key_path, 'wb') as f:
        f.write(encryption_key)

    print(f"Encrypted {file_count} files to {output_path}")
    print(f"Encryption key: {encryption_key.decode()}")
    print(f"Encryption key saved to: {key_path}")
    return output_path, encryption_key

def decrypt_model_for_loading(encrypted_model_path, encryption_key, temp_dir=None):
    """
    Decrypt the model to a temporary directory and return the path to use for loading.
    
    Args:
        encrypted_model_path: Path to the encrypted model
        encryption_key: The key to decrypt the model
        temp_dir: Optional path to use for decryption. If None, a random temp dir is created
    
    Returns:
        Path to the decrypted model that can be passed to load_model
    """
    # Validate the encryption key
    try:
        # Create cipher with the encryption key - this will validate the key format
        cipher = Fernet(encryption_key)
    except Exception as e:
        print(f"Invalid encryption key: {e}")
        return None
    
    # Create a temporary directory for decrypted files if not provided
    if temp_dir is None:
        # Create a unique temp directory that won't conflict
        temp_dir = os.path.join(tempfile.gettempdir(), f"model_decrypt_{uuid.uuid4().hex}")
    
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    print(f"Decrypting model to: {temp_dir}")
    
    # Check if the encrypted model directory exists
    if not os.path.exists(encrypted_model_path):
        print(f"Encrypted model path does not exist: {encrypted_model_path}")
        return None
    
    success_count = 0
    error_count = 0
    
    # Decrypt each file in the encrypted model directory
    for root, dirs, files in os.walk(encrypted_model_path):
        for file in files:
            if file == "encryption_key.bin":  # Skip the key file
                continue
            
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, encrypted_model_path)
            temp_file = os.path.join(temp_dir, rel_path)
            
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(temp_file), exist_ok=True)
            
            try:
                # Read encrypted file
                with open(file_path, 'rb') as f:
                    encrypted_data = f.read()
                
                # Check if this file is actually encrypted
                # A simple heuristic: if it starts with Fernet version identifier (gAAAAA)
                if not encrypted_data.startswith(b'gAAAAA'):
                    print(f"File doesn't appear to be encrypted: {rel_path}")
                    # Copy file as-is
                    with open(temp_file, 'wb') as f:
                        f.write(encrypted_data)
                    success_count += 1
                    continue
                
                # Decrypt file
                try:
                    decrypted_data = cipher.decrypt(encrypted_data)
                    
                    # Save decrypted file temporarily
                    with open(temp_file, 'wb') as f:
                        f.write(decrypted_data)
                    success_count += 1
                except Exception as e:
                    print(f"Error decrypting {rel_path}: {e}")
                    error_count += 1
                    # Try to copy the file as-is as a fallback
                    try:
                        with open(temp_file, 'wb') as f:
                            f.write(encrypted_data)
                        print(f"Copied file without decryption: {rel_path}")
                    except Exception as copy_error:
                        print(f"Failed to copy file: {copy_error}")
            except Exception as e:
                print(f"Error processing {rel_path}: {e}")
                error_count += 1
    
    print(f"Decryption complete. Successfully processed: {success_count}, Errors: {error_count}")
    
    # Check if we have the required tokenizer files
    tokenizer_dir = os.path.join(temp_dir, "tokenizer")
    if not os.path.exists(tokenizer_dir):
        print(f"Warning: Tokenizer directory not found after decryption: {tokenizer_dir}")
        # Try to find or create it
        for root, dirs, files in os.walk(temp_dir):
            if "tokenizer" in dirs:
                print(f"Found tokenizer directory at: {os.path.join(root, 'tokenizer')}")
    
    return temp_dir


# encrypted_model_path = "./encrypted_schnell_model"
# model_path = "/Users/bhuvanpurohit777/.cache/huggingface/hub/models--decompute--schnell-3bit/snapshots/d42f56507000c16a8b57b81875d9afad529a3cd5"
# from cryptography.fernet import Fernet
# encrypted_path, encryption_key = encrypt_model_directory(model_path, "/Users/bhuvanpurohit777/decompute/pdf-viewer-app/encrypted_schnell_model")
