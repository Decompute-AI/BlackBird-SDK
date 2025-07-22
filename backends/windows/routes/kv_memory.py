import sqlite3
import os
import json
from typing import Optional, Dict, List, Any, Union
from datetime import datetime
import threading


class KeyValueMemory:
    """
    A key-value based memory system with SQLite database for permanent storage.
    Supports adding, deleting, retrieving, and managing key-value pairs.
    """
    
    def __init__(self, db_path: str = "memory.db"):
        """
        Initialize the KeyValueMemory system.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.lock = threading.Lock()  # Thread safety
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Create the database and tables if they don't exist."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create the main key-value table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS key_value_store (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_name TEXT UNIQUE NOT NULL,
                    value TEXT NOT NULL,
                    value_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create index for faster key lookups
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_key_name 
                ON key_value_store(key_name)
            ''')
            
            conn.commit()
            conn.close()
    
    def add_data(self, key: str, value: Any, overwrite: bool = True) -> bool:
        """
        Add or update a key-value pair in the memory.
        
        Args:
            key (str): The key to store
            value (Any): The value to store (will be serialized to JSON)
            overwrite (bool): Whether to overwrite existing key
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Serialize value to JSON string
                value_json = json.dumps(value)
                value_type = type(value).__name__
                
                if overwrite:
                    # Use INSERT OR REPLACE for upsert behavior
                    cursor.execute('''
                        INSERT OR REPLACE INTO key_value_store 
                        (key_name, value, value_type, updated_at) 
                        VALUES (?, ?, ?, ?)
                    ''', (key, value_json, value_type, datetime.now()))
                else:
                    # Check if key exists
                    cursor.execute('SELECT key_name FROM key_value_store WHERE key_name = ?', (key,))
                    if cursor.fetchone():
                        return False  # Key already exists and overwrite is False
                    
                    cursor.execute('''
                        INSERT INTO key_value_store 
                        (key_name, value, value_type) 
                        VALUES (?, ?, ?)
                    ''', (key, value_json, value_type))
                
                conn.commit()
                conn.close()
                return True
                
        except Exception as e:
            print(f"Error adding data: {e}")
            return False
    
    def get_data(self, key: str) -> Optional[Any]:
        """
        Retrieve a value by key.
        
        Args:
            key (str): The key to retrieve
            
        Returns:
            Any: The deserialized value, or None if key doesn't exist
        """
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT value, value_type FROM key_value_store 
                    WHERE key_name = ?
                ''', (key,))
                
                result = cursor.fetchone()
                conn.close()
                
                if result:
                    value_json, value_type = result
                    return json.loads(value_json)
                return None
                
        except Exception as e:
            print(f"Error retrieving data: {e}")
            return None
    
    def delete_data(self, key: str) -> bool:
        """
        Delete a key-value pair from memory.
        
        Args:
            key (str): The key to delete
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('DELETE FROM key_value_store WHERE key_name = ?', (key,))
                deleted_count = cursor.rowcount
                
                conn.commit()
                conn.close()
                
                return deleted_count > 0
                
        except Exception as e:
            print(f"Error deleting data: {e}")
            return False
    
    def show_data(self, key: Optional[str] = None, limit: int = 100) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Show data - either a specific key or all keys with pagination.
        
        Args:
            key (str, optional): Specific key to show. If None, shows all keys
            limit (int): Maximum number of records to return when showing all
            
        Returns:
            Union[Dict[str, Any], List[Dict[str, Any]]]: Data in dictionary format
        """
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                if key:
                    # Show specific key
                    cursor.execute('''
                        SELECT key_name, value, value_type, created_at, updated_at 
                        FROM key_value_store WHERE key_name = ?
                    ''', (key,))
                    
                    result = cursor.fetchone()
                    conn.close()
                    
                    if result:
                        key_name, value_json, value_type, created_at, updated_at = result
                        return {
                            'key': key_name,
                            'value': json.loads(value_json),
                            'type': value_type,
                            'created_at': created_at,
                            'updated_at': updated_at
                        }
                    return {}
                else:
                    # Show all keys with limit
                    cursor.execute('''
                        SELECT key_name, value, value_type, created_at, updated_at 
                        FROM key_value_store 
                        ORDER BY updated_at DESC 
                        LIMIT ?
                    ''', (limit,))
                    
                    results = cursor.fetchall()
                    conn.close()
                    
                    data_list = []
                    for row in results:
                        key_name, value_json, value_type, created_at, updated_at = row
                        data_list.append({
                            'key': key_name,
                            'value': json.loads(value_json),
                            'type': value_type,
                            'created_at': created_at,
                            'updated_at': updated_at
                        })
                    
                    return data_list
                    
        except Exception as e:
            print(f"Error showing data: {e}")
            return {} if key else []
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the memory.
        
        Args:
            key (str): The key to check
            
        Returns:
            bool: True if key exists, False otherwise
        """
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('SELECT 1 FROM key_value_store WHERE key_name = ?', (key,))
                result = cursor.fetchone()
                
                conn.close()
                return result is not None
                
        except Exception as e:
            print(f"Error checking key existence: {e}")
            return False
    
    def get_keys(self, pattern: Optional[str] = None) -> List[str]:
        """
        Get all keys, optionally filtered by a pattern.
        
        Args:
            pattern (str, optional): SQL LIKE pattern to filter keys
            
        Returns:
            List[str]: List of keys
        """
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                if pattern:
                    cursor.execute('SELECT key_name FROM key_value_store WHERE key_name LIKE ?', (pattern,))
                else:
                    cursor.execute('SELECT key_name FROM key_value_store')
                
                results = cursor.fetchall()
                conn.close()
                
                return [row[0] for row in results]
                
        except Exception as e:
            print(f"Error getting keys: {e}")
            return []
    
    def clear_all(self) -> bool:
        """
        Clear all data from the memory.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('DELETE FROM key_value_store')
                conn.commit()
                conn.close()
                
                return True
                
        except Exception as e:
            print(f"Error clearing all data: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory store.
        
        Returns:
            Dict[str, Any]: Statistics including total keys, database size, etc.
        """
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Get total count
                cursor.execute('SELECT COUNT(*) FROM key_value_store')
                total_keys = cursor.fetchone()[0]
                
                # Get type distribution
                cursor.execute('''
                    SELECT value_type, COUNT(*) 
                    FROM key_value_store 
                    GROUP BY value_type
                ''')
                type_distribution = dict(cursor.fetchall())
                
                # Get oldest and newest entries
                cursor.execute('''
                    SELECT MIN(created_at), MAX(created_at) 
                    FROM key_value_store
                ''')
                oldest, newest = cursor.fetchone()
                
                conn.close()
                
                # Get database file size
                db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                
                return {
                    'total_keys': total_keys,
                    'type_distribution': type_distribution,
                    'oldest_entry': oldest,
                    'newest_entry': newest,
                    'database_size_bytes': db_size,
                    'database_path': self.db_path
                }
                
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}
    
    def backup_database(self, backup_path: str) -> bool:
        """
        Create a backup of the database.
        
        Args:
            backup_path (str): Path where to save the backup
            
        Returns:
            bool: True if backup successful, False otherwise
        """
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                backup_conn = sqlite3.connect(backup_path)
                
                conn.backup(backup_conn)
                
                conn.close()
                backup_conn.close()
                
                return True
                
        except Exception as e:
            print(f"Error creating backup: {e}")
            return False
    
    def __len__(self) -> int:
        """Return the number of keys in the memory."""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM key_value_store')
                count = cursor.fetchone()[0]
                conn.close()
                return count
        except Exception:
            return 0
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the memory."""
        return self.exists(key)
    
    def __getitem__(self, key: str) -> Any:
        """Get a value by key using dictionary-like syntax."""
        value = self.get_data(key)
        if value is None:
            raise KeyError(f"Key '{key}' not found")
        return value
    
    def __setitem__(self, key: str, value: Any):
        """Set a value by key using dictionary-like syntax."""
        if not self.add_data(key, value):
            raise ValueError(f"Failed to set key '{key}'")






# Example usage and testing
if __name__ == "__main__":
    # Create memory instance
    memory = KeyValueMemory("test_memory.db")
    
    # Add some test data
    print("Adding test data...")
    memory.add_data("user:john", {"name": "John Doe", "age": 30, "city": "New York"})
    memory.add_data("user:jane", {"name": "Jane Smith", "age": 25, "city": "Los Angeles"})
    memory.add_data("config:app", {"version": "1.0.0", "debug": True})
    memory.add_data("counter", 42)
    memory.add_data("message", "Hello, World!")
    
    # Show all data
    print("\nAll data:")
    all_data = memory.show_data()
    for item in all_data:
        print(f"Key: {item['key']}, Value: {item['value']}, Type: {item['type']}")
    
    # Get specific data
    print(f"\nUser John: {memory.get_data('user:john')}")
    print(f"Counter: {memory.get_data('counter')}")
    
    # Check if keys exist
    print(f"\nKey 'user:john' exists: {memory.exists('user:john')}")
    print(f"Key 'nonexistent' exists: {memory.exists('nonexistent')}")
    
    # Get keys with pattern
    print(f"\nUser keys: {memory.get_keys('user:%')}")
    
    # Get statistics
    print(f"\nStatistics: {memory.get_stats()}")
    
    # Dictionary-like access
    print(f"\nUsing dictionary syntax: {memory['user:jane']}")
    
    # Delete a key
    print(f"\nDeleting 'message' key...")
    memory.delete_data("message")
    print(f"Key 'message' exists after deletion: {memory.exists('message')}")
    
    print(f"\nTotal keys in memory: {len(memory)}")






