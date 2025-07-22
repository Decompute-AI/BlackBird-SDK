import platform
import subprocess
import sys
import os
import time
import requests
import psutil
import atexit
import signal
from pathlib import Path
from typing import Optional, Dict, Any, List
from oss_utils.logger import get_logger
import uuid

class PlatformManager:
    """Enhanced platform manager with robust backend process management."""
    _cleanup_registered = False
    def __init__(self, custom_port: Optional[int] = None):
        self.logger = get_logger()
        self.platform = self._detect_platform()
        self.architecture = self._detect_architecture()
        self.backend_process: Optional[subprocess.Popen] = None
        self.backend_port = custom_port or self._find_available_port()
        self.backend_path = self._get_backend_path()
        self.health_check_url = f"http://localhost:{self.backend_port}/health"
        
        self.logger.info(f"Platform Manager initialized for {self.platform}-{self.architecture}")
        self.logger.info(f"Backend path: {self.backend_path}")
        self.logger.info(f"Backend port: {self.backend_port}")
        self._register_cleanup_handlers()

    def _register_cleanup_handlers(self):
        """Register cleanup handlers for graceful shutdown."""
        if not PlatformManager._cleanup_registered:
            PlatformManager._cleanup_registered = True
            atexit.register(self._cleanup_on_exit)
            
            # Register signal handlers
            if hasattr(signal, 'SIGTERM'):
                signal.signal(signal.SIGTERM, self._signal_handler)
            if hasattr(signal, 'SIGINT'):
                signal.signal(signal.SIGINT, self._signal_handler)
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        self.logger.info(f"Received signal {signum}, initiating cleanup...")
        self._cleanup_on_exit()

    def _cleanup_on_exit(self):
        """Cleanup method called on exit."""
        try:
            if self.backend_process:
                self.stop_backend()
            
            # Enhanced cleanup logic that preserves keepalive backends
            self._smart_cleanup_port(self.backend_port)
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def _smart_cleanup_port(self, port: int):
        """Smart cleanup that preserves keepalive backends but kills zombie processes."""
        try:
            # First, check if there's a keepalive backend running
            if self.is_keepalive_backend(port):
                self.logger.info(f"Keepalive backend detected on port {port}, preserving it.")
                # Only cleanup non-keepalive processes
                self._cleanup_non_keepalive_processes(port)
                return
            
            # If no keepalive backend, do full cleanup
            self.logger.info(f"No keepalive backend on port {port}, performing full cleanup.")
            self.force_kill_all_on_port(port)
            
        except Exception as e:
            self.logger.error(f"Error in smart cleanup: {e}")

    def _cleanup_non_keepalive_processes(self, port: int):
        """Clean up only non-keepalive processes on the port."""
        try:
            killed_pids = []
            keepalive_pids = []
            
            # First, identify keepalive backends
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    connections = proc.connections(kind='inet')
                    for conn in connections:
                        if conn.laddr.port == port:
                            # Check if this is a keepalive backend
                            try:
                                resp = requests.get(f"http://localhost:{port}/health", timeout=1)
                                if resp.status_code == 200 and resp.json().get("keepalive", False):
                                    keepalive_pids.append(proc.pid)
                                    self.logger.info(f"Preserving keepalive backend PID {proc.pid}")
                            except Exception:
                                # If health check fails, assume it's not a keepalive backend
                                pass
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            # Kill non-keepalive processes
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    connections = proc.connections(kind='inet')
                    for conn in connections:
                        if conn.laddr.port == port and proc.pid not in keepalive_pids:
                            self.logger.info(f"Killing non-keepalive process {proc.pid} on port {port}")
                            proc.kill()
                            killed_pids.append(proc.pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            if killed_pids:
                self.logger.info(f"Cleaned up {len(killed_pids)} non-keepalive processes: {killed_pids}")
            else:
                self.logger.info("No non-keepalive processes found to clean up")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up non-keepalive processes: {e}")

    def shutdown_keepalive_backend(self, port: int = None):
        """Explicitly shutdown keepalive backend on the specified port."""
        target_port = port or self.backend_port
        try:
            self.logger.info(f"Shutting down keepalive backend on port {target_port}")
            
            # Find and kill keepalive backends
            keepalive_pids = []
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    connections = proc.connections(kind='inet')
                    for conn in connections:
                        if conn.laddr.port == target_port:
                            # Check if this is a keepalive backend
                            try:
                                resp = requests.get(f"http://localhost:{target_port}/health", timeout=1)
                                if resp.status_code == 200 and resp.json().get("keepalive", False):
                                    keepalive_pids.append(proc.pid)
                            except Exception:
                                pass
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            # Kill all keepalive backends
            for pid in keepalive_pids:
                try:
                    p = psutil.Process(pid)
                    self.logger.info(f"Shutting down keepalive backend PID {pid}")
                    p.terminate()
                    try:
                        p.wait(timeout=10)
                        self.logger.info(f"Keepalive backend PID {pid} shut down gracefully")
                    except psutil.TimeoutExpired:
                        self.logger.warning(f"Force killing keepalive backend PID {pid}")
                        p.kill()
                except Exception as e:
                    self.logger.error(f"Error shutting down keepalive backend PID {pid}: {e}")
            
            # Also kill any remaining processes on the port
            self.force_kill_all_on_port(target_port)
            
        except Exception as e:
            self.logger.error(f"Error shutting down keepalive backend: {e}")

    def get_port_status(self, port: int = None) -> dict:
        """Get detailed status of processes on the specified port."""
        target_port = port or self.backend_port
        status = {
            'port': target_port,
            'keepalive_backends': [],
            'other_processes': [],
            'total_processes': 0
        }
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    connections = proc.connections(kind='inet')
                    for conn in connections:
                        if conn.laddr.port == target_port:
                            proc_info = {
                                'pid': proc.pid,
                                'name': proc.info['name'],
                                'cmdline': ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                            }
                            
                            # Check if this is a keepalive backend
                            try:
                                resp = requests.get(f"http://localhost:{target_port}/health", timeout=1)
                                if resp.status_code == 200 and resp.json().get("keepalive", False):
                                    status['keepalive_backends'].append(proc_info)
                                else:
                                    status['other_processes'].append(proc_info)
                            except Exception:
                                status['other_processes'].append(proc_info)
                            
                            status['total_processes'] += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error getting port status: {e}")
        
        return status

    def _detect_platform(self) -> str:
        """Detect the current platform."""
        system = platform.system().lower()
        if system == "darwin":
            return "mac"
        elif system == "windows":
            return "windows" 
        elif system == "linux":
            return "linux"
        else:
            raise RuntimeError(f"Unsupported platform: {system}")
    
    def _detect_architecture(self) -> str:
        """Detect system architecture."""
        machine = platform.machine().lower()
        if machine in ["arm64", "aarch64"]:
            return "arm64"
        elif machine in ["x86_64", "amd64"]:
            return "x86_64"
        else:
            return machine
    
    def _find_available_port(self, start_port: int = 5012) -> int:
        """Find an available port starting from start_port."""
        import socket
        for port in range(start_port, start_port + 100):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('localhost', port))
                    self.logger.info(f"Found available port: {port}")
                    return port
                except OSError:
                    continue
        raise RuntimeError("No available ports found")
    
    def _get_backend_path(self) -> Path:
        """Get platform-specific backend path with enhanced debugging."""
        current_file = Path(__file__).resolve()
        sdk_dir = current_file.parent.parent 
        backends_dir = sdk_dir / "backends"
        
        if self.platform == "mac":
            backend_dir = backends_dir / "mac"
        elif self.platform == "windows":
            backend_dir = backends_dir / "windows"
        elif self.platform == "linux":
            backend_dir = backends_dir / "linux"
        else:
            raise RuntimeError(f"No backend available for {self.platform}")
        
        self.logger.info(f"Selected backend directory: {backend_dir}")
        
        if not backend_dir.exists():
            raise RuntimeError(f"Backend directory not found: {backend_dir}")
        
        backend_script = backend_dir / "decompute.py"
        if not backend_script.exists():
            raise RuntimeError(f"Backend script not found: {backend_script}")
        
        return backend_dir
    
    def is_keepalive_backend(self, port: int) -> bool:
        """Check if the process on the port is a keepalive backend."""
        try:
            import requests
            # First check if anything is listening on the port
            resp = requests.get(f"http://localhost:{port}/health", timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                is_keepalive = data.get("keepalive", False)
                if is_keepalive:
                    self.logger.info(f"Keepalive backend confirmed on port {port}")
                return is_keepalive
        except requests.exceptions.ConnectionError:
            # Port is not listening
            return False
        except Exception as e:
            # Other errors (timeout, etc.)
            self.logger.debug(f"Health check error for port {port}: {e}")
            return False
        return False

    def run_backend_keepalive_async(self):
        """Launch backend in a new terminal with keepalive flag."""
        self.cleanup_keepalive_backends(self.backend_port)
        keepalive_code = str(uuid.uuid4())
        env = os.environ.copy()
        env['BLACKBIRD_KEEPALIVE'] = '1'
        env['BLACKBIRD_KEEPALIVE_CODE'] = keepalive_code
        env['BLACKBIRD_KEEPALIVE_PORT'] = str(self.backend_port)
        
        backend_script = self.backend_path / "decompute.py"
        
        if self.platform == "windows":
            # Create a batch file to run the backend and keep the window open
            batch_content = f'''@echo off
title BlackbirdSDK Keepalive Backend
echo Starting BlackbirdSDK Keepalive Backend...
echo Port: {self.backend_port}
echo Keepalive Code: {keepalive_code}
echo.
"{sys.executable}" "{backend_script}"
echo.
echo Backend stopped. Press any key to close this window...
pause >nul
'''
            # Write batch file
            batch_file = self.backend_path / "run_keepalive.bat"
            with open(batch_file, 'w', encoding='utf-8') as f:
                f.write(batch_content)
            
            # Start the batch file in a new console window
            cmd = [str(batch_file)]
            creationflags = subprocess.CREATE_NEW_CONSOLE
        else:
            # For non-Windows, use a script that keeps the terminal open
            cmd = [sys.executable, str(backend_script)]
            creationflags = 0
        
        # Start the keepalive process
        keepalive_process = subprocess.Popen(
            cmd,
            cwd=str(self.backend_path),
            env=env,
            creationflags=creationflags,
            shell=False  # Don't use shell=True with CREATE_NEW_CONSOLE
        )
        
        self._keepalive_code = keepalive_code
        self._keepalive_process = keepalive_process
        
        self.logger.info(f"Backend keepalive async started with code {keepalive_code} on PID {keepalive_process.pid}")
        
        # Wait a bit for the backend to start
        time.sleep(5)
        
        # Verify the backend is actually running
        max_attempts = 30
        for attempt in range(max_attempts):
            if self.is_keepalive_backend(self.backend_port):
                self.logger.info(f"Keepalive backend confirmed running on port {self.backend_port}")
                break
            time.sleep(2)
            if attempt == max_attempts - 1:
                self.logger.warning(f"Keepalive backend may not have started properly on port {self.backend_port}")
        
        # Register cleanup for this specific keepalive process
        atexit.register(self._cleanup_keepalive_on_exit)
        
        return keepalive_process

    def _cleanup_keepalive_on_exit(self):
        """Cleanup method specifically for keepalive processes."""
        try:
            if hasattr(self, '_keepalive_process') and self._keepalive_process:
                self.logger.info(f"Cleaning up keepalive process PID {self._keepalive_process.pid}")
                try:
                    self._keepalive_process.terminate()
                    self._keepalive_process.wait(timeout=10)
                except (subprocess.TimeoutExpired, psutil.NoSuchProcess):
                    try:
                        self._keepalive_process.kill()
                    except:
                        pass
                finally:
                    self._keepalive_process = None
        except Exception as e:
            self.logger.error(f"Error cleaning up keepalive process: {e}")

    def cleanup_keepalive_backends(self, port: int):
        """Ensure only one keepalive backend is running on the port."""
        import requests
        keepalive_pids = []
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                connections = proc.connections(kind='inet')
                for conn in connections:
                    if conn.laddr.port == port:
                        # Check if this is a keepalive backend
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
                    self.logger.info(f"Killed extra keepalive backend with PID {pid}")
                except Exception:
                    continue
        # If any non-keepalive process is running, kill it
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                connections = proc.connections(kind='inet')
                for conn in connections:
                    if conn.laddr.port == port and proc.pid not in keepalive_pids:
                        proc.kill()
                        self.logger.info(f"Killed non-keepalive process with PID {proc.pid} on port {port}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

    def kill_process_on_port(self, port: int) -> bool:
        """Kill any process using the specified port, unless it's a keepalive backend (leave only one keepalive backend running)."""
        if self.is_keepalive_backend(port):
            self.logger.info(f"Keepalive backend detected on port {port}, not killing.")
            self.cleanup_keepalive_backends(port)
            return False
        killed = False
        try:
            self.logger.info(f"Checking for processes on port {port}")
            
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    connections = proc.connections(kind='inet')
                    for conn in connections:
                        if conn.laddr.port == port:
                            self.logger.info(f"Killing process {proc.info['pid']} ({proc.info['name']}) using port {port}")
                            proc.kill()
                            killed = True
                            time.sleep(2)
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                except Exception as e:
                    self.logger.warning(f"Error checking process: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error killing processes on port {port}: {e}")
        
        return killed
    
    def force_kill_all_on_port(self, port: int):
        """Force kill all processes using the specified port, including keepalive backends."""
        import psutil
        killed_pids = []
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                connections = proc.connections(kind='inet')
                for conn in connections:
                    if conn.laddr.port == port:
                        proc.kill()
                        killed_pids.append(proc.pid)
                        self.logger.info(f"Force killed process {proc.pid} on port {port}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return killed_pids
    
    def start_backend(self, auto_install_deps: bool = True) -> bool:
        """Start platform-specific backend server with proper import path setup."""
        # Check for keepalive backend before any cleanup or error
        if self.is_keepalive_backend(self.backend_port):
            self.logger.info(f"✅ Detected keepalive backend running on port {self.backend_port}. Will reuse it.")
            return True
        
        # Also check if backend is already running (non-keepalive)
        if self.is_backend_running():
            self.logger.info("✅ Backend already running and healthy")
            return True
        
        try:
            # Kill any existing processes
            self.logger.info(f"Cleaning up any existing processes on port {self.backend_port}")
            self.kill_process_on_port(self.backend_port)
            time.sleep(3)
            
            # Check again after cleanup
            if self.is_keepalive_backend(self.backend_port):
                self.logger.info(f"✅ Keepalive backend detected after cleanup on port {self.backend_port}. Will reuse it.")
                return True
            
            if self.is_backend_running():
                self.logger.info("✅ Backend already running and healthy after cleanup")
                return True
            
            backend_script = self.backend_path / "decompute.py"
            
            self.logger.info(f"🚀 Starting new backend server from: {backend_script}")
            self.logger.info(f"Working directory: {self.backend_path}")
            
            # Prepare enhanced environment variables
            env = os.environ.copy()
            
            # CRITICAL: Set PYTHONPATH to include backend directory
            pythonpath_dirs = [
                str(self.backend_path),  # The backend directory itself
                str(self.backend_path.parent),  # The backends directory
                str(self.backend_path.parent.parent),  # The blackbird_sdk directory
            ]
            
            existing_pythonpath = env.get('PYTHONPATH', '')
            if existing_pythonpath:
                pythonpath_dirs.append(existing_pythonpath)
            
            env['PYTHONPATH'] = os.pathsep.join(pythonpath_dirs)
            env['FLASK_ENV'] = 'development'
            env['PYTHONIOENCODING'] = 'utf-8'
            env['FLASK_RUN_PORT'] = str(self.backend_port)
            
            # Log environment setup
            self.logger.info(f"PYTHONPATH set to: {env['PYTHONPATH']}")
            if self.platform=="mac":
                cmd= [sys.executable, str(backend_script), "run_server"]
            else:
                cmd = [sys.executable, str(backend_script)]
            
            self.logger.info(f"Starting backend with command: {' '.join(cmd)}")
            
            # Start backend process with enhanced configuration
            self.backend_process = subprocess.Popen(
                cmd,
                cwd=str(self.backend_path),  # CRITICAL: Set working directory
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                stdin=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                universal_newlines=True
            )
            
            self.logger.info(f"Backend process started with PID: {self.backend_process.pid}")
            
            # Wait for server with real-time output monitoring
            if self._wait_for_backend_with_output(timeout=180):
                self.logger.info(f"✅ Backend server started successfully on port {self.backend_port}")
                return True
            else:
                self.logger.error("❌ Backend server failed to start within timeout")
                self.stop_backend()
                return False
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start backend: {e}")
            if self.backend_process:
                self.stop_backend()
            return False
        

    def _wait_for_backend_with_output(self, timeout: int = 180) -> bool:
        """Wait for backend with real-time output monitoring."""
        start_time = time.time()
        self.logger.info(f"Waiting for backend to be ready (timeout: {timeout}s)")
        
        # Monitor output in real-time
        import threading
        
        def read_output():
            if self.backend_process and self.backend_process.stdout:
                for line in iter(self.backend_process.stdout.readline, ''):
                    if line:
                        self.logger.info(f"Backend: {line.strip()}")
        
        output_thread = threading.Thread(target=read_output, daemon=True)
        output_thread.start()
        
        # Wait for process to stabilize
        time.sleep(15)
        
        while time.time() - start_time < timeout:
            elapsed = time.time() - start_time
            
            # Check if process died
            if self.backend_process.poll() is not None:
                self.logger.error(f"Backend process died after {elapsed:.1f}s")
                return False
            
            # Check if server is responding
            try:
                response = requests.get(self.health_check_url, timeout=10)
                if response.status_code == 200:
                    self.logger.info(f"Backend health check successful after {elapsed:.1f}s")
                    time.sleep(5)  # Additional stabilization
                    return True
            except requests.exceptions.ConnectionError:
                pass
            except Exception as e:
                self.logger.warning(f"Health check error: {e}")
            
            # Progress logging
            if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                self.logger.info(f"Still waiting for backend... ({elapsed:.0f}s elapsed)")
            
            time.sleep(2)
        
        self.logger.error(f"Backend failed to start within {timeout}s")
        return False
    
    def _log_backend_output(self):
        """Log backend process output for debugging."""
        if self.backend_process:
            try:
                stdout_data = ""
                stderr_data = ""
                
                if self.backend_process.stdout:
                    try:
                        # Use communicate with timeout to get output without blocking
                        stdout_data, stderr_data = self.backend_process.communicate(timeout=1)
                    except subprocess.TimeoutExpired:
                        # Process is still running, get partial output
                        try:
                            stdout_data = self.backend_process.stdout.read()
                        except:
                            pass
                        try:
                            stderr_data = self.backend_process.stderr.read()
                        except:
                            pass
                    except Exception as e:
                        # Handle interpreter shutdown gracefully
                        if "can't create new thread" in str(e) or "interpreter shutdown" in str(e):
                            self.logger.info("Skipping backend output read during interpreter shutdown")
                            return
                        else:
                            raise
                
                if stdout_data and stdout_data.strip():
                    self.logger.info(f"Backend STDOUT: {stdout_data}")
                if stderr_data and stderr_data.strip():
                    self.logger.error(f"Backend STDERR: {stderr_data}")
                    
            except Exception as e:
                # Handle interpreter shutdown gracefully
                if "can't create new thread" in str(e) or "interpreter shutdown" in str(e):
                    self.logger.info("Skipping backend output read during interpreter shutdown")
                else:
                    self.logger.error(f"Failed to read backend output: {e}")
    
    def stop_backend(self):
        """Stop backend server with comprehensive cleanup."""
        if self.backend_process:
            try:
                self.logger.info(f"Stopping backend process (PID: {self.backend_process.pid})")
                
                # Try graceful shutdown first
                self.backend_process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.backend_process.wait(timeout=15)
                    self.logger.info("Backend stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown failed
                    self.logger.warning("Backend didn't stop gracefully, force killing")
                    self.backend_process.kill()
                    try:
                        self.backend_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.logger.error("Failed to kill backend process")
                
                # Log final output
                self._log_backend_output()
                
            except Exception as e:
                self.logger.error(f"Error stopping backend: {e}")
            finally:
                self.backend_process = None
                
        # Use force kill for more aggressive cleanup
        self.logger.info(f"Force cleaning up port {self.backend_port}")
        self.force_kill_all_on_port(self.backend_port)
    
    def is_backend_running(self) -> bool:
        """Check if backend server is running and responsive."""
        if not self.backend_process or self.backend_process.poll() is not None:
            return False
        
        try:
            response = requests.get(self.health_check_url, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def restart_backend(self) -> bool:
        """Restart the backend server."""
        self.logger.info("Restarting backend server...")
        self.stop_backend()
        time.sleep(5)
        return self.start_backend()
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get comprehensive information about current backend."""
        info = {
            "platform": self.platform,
            "architecture": self.architecture,
            "backend_path": str(self.backend_path),
            "port": self.backend_port,
            "running": self.is_backend_running(),
            "health_url": self.health_check_url
        }
        
        if self.backend_process:
            info.update({
                "process_id": self.backend_process.pid,
                "process_status": "running" if self.backend_process.poll() is None else "stopped"
            })
        
        return info
    
    def get_backend_url(self) -> str:
        """Get the base URL for the backend."""
        return f"http://localhost:{self.backend_port}"
    
    def __enter__(self):
        """Context manager entry."""
        self.start_backend()
        return self
    def find_backend_process(self, port: int) -> Optional[int]:
        """Find existing backend process using the port (FIXED for psutil 6.0+)."""
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    # Use net_connections() for newer psutil versions (6.0+)
                    try:
                        connections = proc.net_connections(kind='inet')
                    except (AttributeError, psutil.AccessDenied):
                        # Fall back to connections() for older versions
                        try:
                            connections = proc.connections(kind='inet')
                        except (AttributeError, psutil.AccessDenied):
                            continue
                    
                    if connections:
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
            self.logger.warning(f"Error finding backend process: {e}")
        return None

    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_backend()
