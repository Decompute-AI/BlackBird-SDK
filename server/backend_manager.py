"""
Enhanced backend manager with port conflict prevention.
"""

import threading
import socket
import time
import psutil
import subprocess
import os
import atexit
import requests
from typing import Optional

class BackendManager:
    """Thread-safe singleton manager for the backend process (OSS version)."""
    _instance: Optional['BackendManager'] = None
    _lock = threading.RLock()
    _initialization_lock = threading.Lock()
    _cleanup_registered = False

    def __init__(self):
        self._backend_port = 5012
        self._backend_process = None
        self._backend_process_id = None
        self._is_initializing = False
        self._backend_script = os.path.abspath(os.path.join(os.path.dirname(__file__), '../backends/windows/decompute.py'))
        self._python_executable = r"C:\decompute-app\.venv\Scripts\python.exe"
        self._register_global_cleanup()

    @classmethod
    def get_instance(cls) -> 'BackendManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = BackendManager()
        return cls._instance

    @classmethod
    def _register_global_cleanup(cls):
        if not cls._cleanup_registered:
            cls._cleanup_registered = True
            atexit.register(cls._cleanup_all_instances)

    @classmethod
    def _cleanup_all_instances(cls):
        try:
            if cls._instance is not None:
                with cls._lock:
                    if cls._instance is not None:
                        cls._instance.logger.info("Cleaning up BackendManager instance on exit")
                        
                        # Stop backend if running
                        if cls._instance._backend_process and cls._instance._backend_process.poll() is None:
                            cls._instance.stop()
                        
                        # Force cleanup port
                        cls._instance._force_cleanup_port(cls._instance._backend_port)
                        
                        # Clear instance
                        cls._instance = None
        except Exception as e:
            print(f"Error during BackendManager cleanup: {e}")

    def is_keepalive_backend(self, port: int) -> bool:
        """Check if the process on the port is a keepalive backend."""
        try:
            resp = requests.get(f"http://localhost:{port}/health", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("keepalive", False)
        except Exception:
            pass
        return False

    def cleanup_keepalive_backends(self, port: int):
        """Ensure only one keepalive backend is running on the port."""
        keepalive_pids = []
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                connections = proc.connections(kind='inet')
                for conn in connections:
                    if conn.laddr.port == port:
                        try:
                            resp = requests.get(f"http://localhost:{port}/health", timeout=1)
                            if resp.status_code == 200 and resp.json().get("keepalive", False):
                                keepalive_pids.append(proc.pid)
                        except Exception:
                            continue
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        # If more than one keepalive backend, kill extras
        if len(keepalive_pids) > 1:
            for pid in keepalive_pids[1:]:
                try:
                    p = psutil.Process(pid)
                    p.kill()
                except Exception:
                    continue
        # If any non-keepalive process is running, kill it
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                connections = proc.connections(kind='inet')
                for conn in connections:
                    if conn.laddr.port == port and proc.pid not in keepalive_pids:
                        proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

    def is_port_available(self, port: int) -> bool:
        """Check if port is available for binding."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1.0)
                result = sock.connect_ex(('localhost', port))
                return result != 0  # Port is available if connection fails
        except Exception:
            return False

    def find_backend_process(self, port: int) -> Optional[int]:
        """Find existing backend process using the port."""
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    # Use the fixed psutil method from platform_manager
                    try:
                        connections = proc.net_connections(kind='inet')
                    except (AttributeError, psutil.AccessDenied):
                        try:
                            connections = proc.connections(kind='inet')
                        except (AttributeError, psutil.AccessDenied):
                            continue
                    for conn in connections:
                        if (hasattr(conn, 'laddr') and conn.laddr and 
                            hasattr(conn.laddr, 'port') and 
                            conn.laddr.port == port and
                            hasattr(conn, 'status') and
                            conn.status == psutil.CONN_LISTEN):
                            return proc.info['pid']
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except Exception as e:
            print(f"Error finding backend process: {e}")
        return None

    def _force_cleanup_port(self, port: int) -> bool:
        """Force cleanup of processes on specified port, respecting keepalive backends."""
        self.cleanup_keepalive_backends(port)
        if self.is_keepalive_backend(port):
            print(f"Keepalive backend detected on port {port}, not killing.")
            return False
        try:
            print(f"Force cleaning up port {port}")
            # Kill processes using the port
            killed_any = False
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    # Use the fixed psutil method from platform_manager
                    try:
                        connections = proc.net_connections(kind='inet')
                    except (AttributeError, psutil.AccessDenied):
                        try:
                            connections = proc.connections(kind='inet')
                        except (AttributeError, psutil.AccessDenied):
                            continue
                    for conn in connections:
                        if (hasattr(conn, 'laddr') and conn.laddr and 
                            hasattr(conn.laddr, 'port') and 
                            conn.laddr.port == port):
                            print(f"Force killing process {proc.info['pid']} on port {port}")
                            proc.kill()
                            killed_any = True
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            if killed_any:
                time.sleep(2)  # Allow time for cleanup
            return True
        except Exception as e:
            print(f"Error in force cleanup: {e}")
            return False

    def _verify_backend_health(self, port: int) -> bool:
        """Verify backend is healthy and responding."""
        try:
            health_url = f"http://localhost:{port}/health"
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def start(self, port: Optional[int] = None) -> bool:
        """Start backend with comprehensive conflict prevention."""
        target_port = port or self._backend_port
        
        # PATCH: Check for keepalive backend before any cleanup or error
        if self.is_keepalive_backend(target_port):
            print(f"Detected keepalive backend running on port {target_port}. Will reuse it.")
            return True
        
        with self._initialization_lock:
            # Prevent multiple simultaneous initialization attempts
            if self._is_initializing:
                print("Backend initialization already in progress")
                return self._wait_for_initialization()
            
            self._is_initializing = True
            
            try:
                # Force cleanup any existing processes first
                if not self.is_port_available(target_port):
                    print(f"Port {target_port} is occupied, forcing cleanup...")
                    self._force_cleanup_port(target_port)
                    time.sleep(2)
                
                # Check if backend is already running on this manager instance
                if self._backend_process and self._backend_process.poll() is None:
                    print("Backend already running on this manager")
                    return True
                
                # Check if port is available
                if not self.is_port_available(target_port):
                    existing_pid = self.find_backend_process(target_port)
                    if existing_pid:
                        print(f"Backend already running on port {target_port} (PID: {existing_pid})")
                        # Try to reuse existing backend
                        return self._verify_backend_health(target_port)
                    else:
                        print(f"Port {target_port} is occupied by unknown process")
                        return False
                
                # Start backend
                if not os.path.exists(self._backend_script):
                    print(f"Backend script not found: {self._backend_script}")
                    return False
                print(f"[BackendManager] Starting backend: {self._backend_script} on port {target_port}")
                self._backend_process = subprocess.Popen([
                    self._python_executable, self._backend_script, str(target_port)
                ])
                time.sleep(3)
                
                # Verify backend is actually running
                if self._verify_backend_health(target_port):
                    self._backend_process_id = self.find_backend_process(target_port)
                    print(f"Backend started successfully on port {target_port} (PID: {self._backend_process_id})")
                    return True
                else:
                    print("Backend health check failed after startup")
                    return False
                    
            except Exception as e:
                print(f"Error starting backend: {e}")
                return False
            finally:
                self._is_initializing = False

    def _wait_for_initialization(self) -> bool:
        """Wait for ongoing initialization to complete."""
        max_wait = 30  # seconds
        wait_interval = 0.5
        elapsed = 0
        
        while self._is_initializing and elapsed < max_wait:
            time.sleep(wait_interval)
            elapsed += wait_interval
        
        # Check if backend is now running
        return self._backend_process and self._backend_process.poll() is None

    def stop(self) -> bool:
        """Stop the backend server."""
        with self._lock:
            if not self._backend_process or self._backend_process.poll() is not None:
                print("Backend is not running")
                self._force_cleanup_port(self._backend_port)
                return True

            print("Stopping backend server...")
            try:
                self._backend_process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self._backend_process.wait(timeout=15)
                    print("Backend stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown failed
                    print("Backend didn't stop gracefully, force killing")
                    self._backend_process.kill()
                    try:
                        self._backend_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print("Failed to kill backend process")
                
                self._backend_process = None
                
            except Exception as e:
                print(f"Error stopping backend: {e}")
                return False
                
            # Also kill any remaining processes on the port
            self._force_cleanup_port(self._backend_port)
            return True

    def get_backend_status(self) -> dict:
        """Get comprehensive backend status."""
        health_check = self._verify_backend_health(self._backend_port)
        is_keepalive = self.is_keepalive_backend(self._backend_port)
        is_running = (self._backend_process and self._backend_process.poll() is None) or (is_keepalive and health_check)
        return {
            'is_running': is_running,
            'port': self._backend_port,
            'process_id': self._backend_process_id,
            'port_available': self.is_port_available(self._backend_port),
            'health_check': health_check
        }

# Public API functions

def start_backend_server(port: Optional[int] = None) -> bool:
    """Public function to start the backend."""
    return BackendManager.get_instance().start(port)

def stop_backend_server() -> bool:
    """Public function to stop the backend."""
    return BackendManager.get_instance().stop()

def get_backend_status() -> dict:
    """Get backend status information."""
    return BackendManager.get_instance().get_backend_status()
