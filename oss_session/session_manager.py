"""SessionManager for handling user sessions, quotas, and rate limiting."""

import threading
import time
import json
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque

from .session_types import (
    SessionData, SessionStatus, QuotaType, QuotaLimit, RateLimit, RateLimitType,
    UsageRecord, SessionQuotaPresets, SessionRateLimitPresets
)
from oss_utils.errors import ValidationError, QuotaExceededError, RateLimitError
from oss_utils.feature_flags import require_feature, is_feature_enabled
from oss_utils.logger import get_logger

class QuotaExceededError(Exception):
    """Raised when quota limits are exceeded."""
    def __init__(self, message: str, quota_type: QuotaType, used: int, limit: int):
        super().__init__(message)
        self.quota_type = quota_type
        self.used = used
        self.limit = limit

class RateLimitError(Exception):
    """Raised when rate limits are exceeded."""
    def __init__(self, message: str, limit_type: RateLimitType, retry_after: float = None):
        super().__init__(message)
        self.limit_type = limit_type
        self.retry_after = retry_after

class SessionManager:
    """Manages user sessions, quotas, usage tracking, and rate limiting."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the SessionManager."""
        self.logger = get_logger()
        self.config = config or {}
        
        # Session storage
        self.sessions: Dict[str, SessionData] = {}
        self.user_sessions: Dict[str, List[str]] = defaultdict(list)  # user_id -> [session_ids]
        
        # Usage tracking
        self.usage_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.usage_callbacks: List[Callable] = []
        
        # Rate limiting
        self.rate_limit_windows: Dict[str, Dict[RateLimitType, RateLimit]] = defaultdict(dict)
        
        # Configuration
        self.session_timeout = self.config.get('session_timeout', 3600)  # 1 hour default
        self.max_sessions_per_user = self.config.get('max_sessions_per_user', 10)
        self.cleanup_interval = self.config.get('cleanup_interval', 300)  # 5 minutes
        
        # Background tasks
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        self._lock = threading.RLock()
        
        # Start background cleanup
        self._start_cleanup_thread()
        
        self.logger.info("SessionManager initialized", 
                        session_timeout=self.session_timeout,
                        max_sessions_per_user=self.max_sessions_per_user)
    
    def create_session(self, user_id: str, tier: str = "free", 
                      metadata: Dict[str, Any] = None) -> SessionData:
        """Create a new user session with appropriate quotas."""
        if not user_id:
            raise ValidationError("User ID is required", field_name="user_id")
        
        with self._lock:
            # Check max sessions per user
            if len(self.user_sessions[user_id]) >= self.max_sessions_per_user:
                # Clean up oldest inactive session
                self._cleanup_user_sessions(user_id)
                
                # If still at limit, raise error
                if len(self.user_sessions[user_id]) >= self.max_sessions_per_user:
                    raise ValidationError(
                        f"Maximum sessions per user exceeded: {self.max_sessions_per_user}",
                        field_name="sessions"
                    )
            
            # Generate session ID
            import uuid
            session_id = str(uuid.uuid4())
            
            # Get quota configuration based on tier
            quotas = self._get_tier_quotas(tier)
            rate_limits = self._get_tier_rate_limits(tier)
            
            # Create session
            session = SessionData(
                session_id=session_id,
                user_id=user_id,
                created_at=time.time(),
                last_activity=time.time(),
                status=SessionStatus.ACTIVE,
                quotas=quotas,
                rate_limits=rate_limits,
                metadata=metadata or {}
            )
            
            # Store session
            self.sessions[session_id] = session
            self.user_sessions[user_id].append(session_id)
            
            self.logger.info("Session created", 
                           session_id=session_id,
                           user_id=user_id,
                           tier=tier)
            
            return session
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session by ID."""
        with self._lock:
            session = self.sessions.get(session_id)
            if session and session.is_active:
                session.update_activity()
                return session
            return None
    
    def get_user_sessions(self, user_id: str, active_only: bool = True) -> List[SessionData]:
        """Get all sessions for a user."""
        with self._lock:
            session_ids = self.user_sessions.get(user_id, [])
            sessions = []
            
            for session_id in session_ids:
                if session_id in self.sessions:
                    session = self.sessions[session_id]
                    if not active_only or session.is_active:
                        sessions.append(session)
            
            return sessions
    
    def update_session_activity(self, session_id: str) -> bool:
        """Update session activity timestamp."""
        with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id].update_activity()
                return True
            return False
    
    def terminate_session(self, session_id: str) -> bool:
        """Terminate a specific session."""
        with self._lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                session.status = SessionStatus.TERMINATED
                
                # Remove from user sessions list
                self.user_sessions[session.user_id] = [
                    sid for sid in self.user_sessions[session.user_id] 
                    if sid != session_id
                ]
                
                # Remove from storage
                del self.sessions[session_id]
                
                self.logger.info("Session terminated", session_id=session_id)
                return True
            return False
    
    def terminate_user_sessions(self, user_id: str) -> int:
        """Terminate all sessions for a user."""
        with self._lock:
            session_ids = self.user_sessions.get(user_id, []).copy()
            terminated_count = 0
            
            for session_id in session_ids:
                if self.terminate_session(session_id):
                    terminated_count += 1
            
            self.logger.info("User sessions terminated", 
                           user_id=user_id,
                           count=terminated_count)
            
            return terminated_count
    
    def get_quota(self, session_id: str, quota_type: QuotaType = None) -> Dict[str, Any]:
        """Get quota information for a session."""
        session = self.get_session(session_id)
        if not session:
            raise ValidationError("Session not found", field_name="session_id")
        
        with self._lock:
            # Reset quotas if needed
            for quota in session.quotas.values():
                quota.reset_if_needed()
            
            if quota_type:
                quota = session.quotas.get(quota_type)
                if quota:
                    return {
                        'type': quota_type.value,
                        'limit': quota.limit,
                        'used': quota.used,
                        'remaining': quota.remaining,
                        'usage_percentage': quota.usage_percentage,
                        'reset_period': quota.reset_period,
                        'last_reset': quota.last_reset
                    }
                else:
                    return {'error': f'Quota type {quota_type.value} not found'}
            
            # Return all quotas
            quota_info = {}
            for qtype, quota in session.quotas.items():
                quota_info[qtype.value] = {
                    'limit': quota.limit,
                    'used': quota.used,
                    'remaining': quota.remaining,
                    'usage_percentage': quota.usage_percentage,
                    'reset_period': quota.reset_period,
                    'last_reset': quota.last_reset
                }
            
            return quota_info
    
    def track_usage(self, session_id: str, operation: str, 
                   resource_type: QuotaType, amount: int, 
                   metadata: Dict[str, Any] = None) -> bool:
        """Track resource usage and update quotas."""
        session = self.get_session(session_id)
        if not session:
            raise ValidationError("Session not found", field_name="session_id")
        
        with self._lock:
            # Check and update quota
            if resource_type in session.quotas:
                quota = session.quotas[resource_type]
                quota.reset_if_needed()
                
                # Check if this would exceed quota
                if quota.used + amount > quota.limit:
                    raise QuotaExceededError(
                        f"Quota exceeded for {resource_type.value}",
                        resource_type,
                        quota.used + amount,
                        quota.limit
                    )
                
                # Update quota
                quota.used += amount
            
            # Record usage
            usage_record = UsageRecord.create(
                user_id=session.user_id,
                session_id=session_id,
                operation=operation,
                resource_type=resource_type,
                amount=amount,
                **(metadata or {})
            )
            
            self.usage_history[session.user_id].append(usage_record)
            
            # Call usage callbacks
            for callback in self.usage_callbacks:
                try:
                    callback(usage_record)
                except Exception as e:
                    self.logger.error("Error in usage callback", error=str(e))
            
            session.update_activity()
            
            self.logger.debug("Usage tracked", 
                            session_id=session_id,
                            operation=operation,
                            resource_type=resource_type.value,
                            amount=amount)
            
            return True
    
    def enforce_rate_limit(self, session_id: str, operation: str, 
                          limit_type: RateLimitType = RateLimitType.PER_MINUTE) -> bool:
        """Enforce rate limiting for an operation."""
        session = self.get_session(session_id)
        if not session:
            raise ValidationError("Session not found", field_name="session_id")
        
        with self._lock:
            if limit_type in session.rate_limits:
                rate_limit = session.rate_limits[limit_type]
                
                if rate_limit.is_exceeded():
                    # Calculate retry after time
                    retry_after = rate_limit.window_size - (time.time() - rate_limit.window_start)
                    
                    raise RateLimitError(
                        f"Rate limit exceeded for {limit_type.value}",
                        limit_type,
                        retry_after=max(0, retry_after)
                    )
                
                # Increment rate limit counter
                if not rate_limit.increment():
                    retry_after = rate_limit.window_size - (time.time() - rate_limit.window_start)
                    raise RateLimitError(
                        f"Rate limit exceeded for {limit_type.value}",
                        limit_type,
                        retry_after=max(0, retry_after)
                    )
            
            session.update_activity()
            return True
    
    def manage_concurrency(self, session_id: str, operation: str, 
                          increment: bool = True) -> bool:
        """Manage concurrent operation limits."""
        session = self.get_session(session_id)
        if not session:
            raise ValidationError("Session not found", field_name="session_id")
        
        with self._lock:
            if increment:
                if session.concurrent_operations >= session.max_concurrent_operations:
                    raise RateLimitError(
                        f"Maximum concurrent operations exceeded: {session.max_concurrent_operations}",
                        RateLimitType.CONCURRENT
                    )
                session.concurrent_operations += 1
            else:
                session.concurrent_operations = max(0, session.concurrent_operations - 1)
            
            session.update_activity()
            
            self.logger.debug("Concurrency managed", 
                            session_id=session_id,
                            operation=operation,
                            increment=increment,
                            current_count=session.concurrent_operations)
            
            return True
    
    def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get complete session data."""
        session = self.get_session(session_id)
        if not session:
            return None
        
        with self._lock:
            return {
                'session_id': session.session_id,
                'user_id': session.user_id,
                'status': session.status.value,
                'created_at': session.created_at,
                'last_activity': session.last_activity,
                'age_seconds': session.age_seconds,
                'idle_seconds': session.idle_seconds,
                'concurrent_operations': session.concurrent_operations,
                'max_concurrent_operations': session.max_concurrent_operations,
                'quotas': self.get_quota(session_id),
                'metadata': session.metadata
            }
    
    def get_usage_history(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get usage history for a user."""
        with self._lock:
            usage_records = list(self.usage_history.get(user_id, []))[-limit:]
            
            return [
                {
                    'timestamp': record.timestamp,
                    'operation': record.operation,
                    'resource_type': record.resource_type.value,
                    'amount': record.amount,
                    'metadata': record.metadata
                }
                for record in usage_records
            ]
    
    def add_usage_callback(self, callback: Callable[[UsageRecord], None]):
        """Add a usage tracking callback."""
        self.usage_callbacks.append(callback)
    
    def remove_usage_callback(self, callback: Callable[[UsageRecord], None]):
        """Remove a usage tracking callback."""
        if callback in self.usage_callbacks:
            self.usage_callbacks.remove(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        with self._lock:
            active_sessions = sum(1 for s in self.sessions.values() if s.is_active)
            total_users = len(self.user_sessions)
            
            return {
                'total_sessions': len(self.sessions),
                'active_sessions': active_sessions,
                'total_users': total_users,
                'average_sessions_per_user': len(self.sessions) / max(1, total_users),
                'session_timeout': self.session_timeout,
                'max_sessions_per_user': self.max_sessions_per_user
            }
    
    def _get_tier_quotas(self, tier: str) -> Dict[QuotaType, QuotaLimit]:
        """Get quotas for a user tier."""
        tier_methods = {
            'free': SessionQuotaPresets.free_tier,
            'basic': SessionQuotaPresets.basic_tier,
            'pro': SessionQuotaPresets.pro_tier,
            'enterprise': SessionQuotaPresets.enterprise_tier
        }
        
        method = tier_methods.get(tier, SessionQuotaPresets.free_tier)
        return method()
    
    def _get_tier_rate_limits(self, tier: str) -> Dict[RateLimitType, RateLimit]:
        """Get rate limits for a user tier."""
        if tier in ['pro', 'enterprise']:
            return SessionRateLimitPresets.premium_limits()
        else:
            return SessionRateLimitPresets.default_limits()
    
    def _cleanup_user_sessions(self, user_id: str):
        """Clean up inactive sessions for a user."""
        session_ids = self.user_sessions.get(user_id, []).copy()
        
        for session_id in session_ids:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                if not session.is_active or session.idle_seconds > self.session_timeout:
                    self.terminate_session(session_id)
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self._cleanup_thread.start()
    
    def _cleanup_loop(self):
        """Background cleanup loop."""
        while not self._stop_cleanup.is_set():
            try:
                self._cleanup_expired_sessions()
                self._stop_cleanup.wait(self.cleanup_interval)
            except Exception as e:
                self.logger.error("Error in cleanup loop", error=str(e))
                time.sleep(5)
    
    def _cleanup_expired_sessions(self):
        """Clean up expired and inactive sessions."""
        current_time = time.time()
        expired_sessions = []
        
        with self._lock:
            for session_id, session in self.sessions.items():
                if (session.idle_seconds > self.session_timeout or 
                    session.status != SessionStatus.ACTIVE):
                    expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.terminate_session(session_id)
        
        if expired_sessions:
            self.logger.info("Cleaned up expired sessions", count=len(expired_sessions))
    
    def cleanup(self):
        """Clean up resources."""
        self.logger.info("Cleaning up SessionManager")
        
        # Stop cleanup thread
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        
        # Clear all sessions
        with self._lock:
            self.sessions.clear()
            self.user_sessions.clear()
            self.usage_history.clear()
            self.usage_callbacks.clear()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass
