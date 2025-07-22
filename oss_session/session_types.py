"""Session types and configurations for the Decompute SDK."""

import time
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

class SessionStatus(Enum):
    """Status of a user session."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"

class QuotaType(Enum):
    """Types of usage quotas."""
    TOKENS = "tokens"
    REQUESTS = "requests"
    STORAGE = "storage"
    COMPUTE_TIME = "compute_time"
    FILE_UPLOADS = "file_uploads"
    TRAINING_JOBS = "training_jobs"

class RateLimitType(Enum):
    """Types of rate limits."""
    PER_SECOND = "per_second"
    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"
    PER_DAY = "per_day"
    CONCURRENT = "concurrent"

@dataclass
class QuotaLimit:
    """Represents a quota limit for a specific resource."""
    quota_type: QuotaType
    limit: int
    used: int = 0
    reset_period: str = "monthly"  # daily, weekly, monthly
    last_reset: float = field(default_factory=time.time)
    
    @property
    def remaining(self) -> int:
        """Get remaining quota."""
        return max(0, self.limit - self.used)
    
    @property
    def usage_percentage(self) -> float:
        """Get usage percentage."""
        if self.limit == 0:
            return 0.0
        return min(100.0, (self.used / self.limit) * 100)
    
    def is_exceeded(self) -> bool:
        """Check if quota is exceeded."""
        return self.used >= self.limit
    
    def should_reset(self) -> bool:
        """Check if quota should be reset based on period."""
        current_time = time.time()
        
        if self.reset_period == "daily":
            return current_time - self.last_reset >= 86400  # 24 hours
        elif self.reset_period == "weekly":
            return current_time - self.last_reset >= 604800  # 7 days
        elif self.reset_period == "monthly":
            return current_time - self.last_reset >= 2592000  # 30 days
        
        return False
    
    def reset_if_needed(self):
        """Reset quota if reset period has passed."""
        if self.should_reset():
            self.used = 0
            self.last_reset = time.time()

@dataclass
class RateLimit:
    """Represents a rate limit configuration."""
    limit_type: RateLimitType
    limit: int
    window_size: int  # seconds
    current_count: int = 0
    window_start: float = field(default_factory=time.time)
    
    def is_exceeded(self) -> bool:
        """Check if rate limit is exceeded."""
        self._reset_window_if_needed()
        return self.current_count >= self.limit
    
    def increment(self) -> bool:
        """Increment counter and check if limit is exceeded."""
        self._reset_window_if_needed()
        self.current_count += 1
        return self.current_count <= self.limit
    
    def _reset_window_if_needed(self):
        """Reset window if time has passed."""
        current_time = time.time()
        if current_time - self.window_start >= self.window_size:
            self.current_count = 0
            self.window_start = current_time

@dataclass
class SessionData:
    """User session data."""
    session_id: str
    user_id: str
    created_at: float
    last_activity: float
    status: SessionStatus
    quotas: Dict[QuotaType, QuotaLimit] = field(default_factory=dict)
    rate_limits: Dict[RateLimitType, RateLimit] = field(default_factory=dict)
    concurrent_operations: int = 0
    max_concurrent_operations: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self.status == SessionStatus.ACTIVE
    
    @property
    def age_seconds(self) -> float:
        """Get session age in seconds."""
        return time.time() - self.created_at
    
    @property
    def idle_seconds(self) -> float:
        """Get idle time in seconds."""
        return time.time() - self.last_activity
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()

@dataclass
class UsageRecord:
    """Records a usage event."""
    timestamp: float
    user_id: str
    session_id: str
    operation: str
    resource_type: QuotaType
    amount: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(cls, user_id: str, session_id: str, operation: str, 
               resource_type: QuotaType, amount: int, **metadata):
        """Create a new usage record."""
        return cls(
            timestamp=time.time(),
            user_id=user_id,
            session_id=session_id,
            operation=operation,
            resource_type=resource_type,
            amount=amount,
            metadata=metadata
        )

class SessionQuotaPresets:
    """Predefined quota configurations for different user tiers."""
    
    @staticmethod
    def free_tier() -> Dict[QuotaType, QuotaLimit]:
        """Free tier quota limits."""
        return {
            QuotaType.TOKENS: QuotaLimit(QuotaType.TOKENS, 10000, reset_period="daily"),
            QuotaType.REQUESTS: QuotaLimit(QuotaType.REQUESTS, 100, reset_period="daily"),
            QuotaType.STORAGE: QuotaLimit(QuotaType.STORAGE, 100 * 1024 * 1024, reset_period="monthly"),  # 100MB
            QuotaType.FILE_UPLOADS: QuotaLimit(QuotaType.FILE_UPLOADS, 10, reset_period="daily"),
            QuotaType.TRAINING_JOBS: QuotaLimit(QuotaType.TRAINING_JOBS, 0, reset_period="monthly"),
        }
    
    @staticmethod
    def basic_tier() -> Dict[QuotaType, QuotaLimit]:
        """Basic tier quota limits."""
        return {
            QuotaType.TOKENS: QuotaLimit(QuotaType.TOKENS, 100000, reset_period="monthly"),
            QuotaType.REQUESTS: QuotaLimit(QuotaType.REQUESTS, 1000, reset_period="daily"),
            QuotaType.STORAGE: QuotaLimit(QuotaType.STORAGE, 1024 * 1024 * 1024, reset_period="monthly"),  # 1GB
            QuotaType.FILE_UPLOADS: QuotaLimit(QuotaType.FILE_UPLOADS, 50, reset_period="daily"),
            QuotaType.TRAINING_JOBS: QuotaLimit(QuotaType.TRAINING_JOBS, 5, reset_period="monthly"),
        }
    
    @staticmethod
    def pro_tier() -> Dict[QuotaType, QuotaLimit]:
        """Pro tier quota limits."""
        return {
            QuotaType.TOKENS: QuotaLimit(QuotaType.TOKENS, 1000000, reset_period="monthly"),
            QuotaType.REQUESTS: QuotaLimit(QuotaType.REQUESTS, 10000, reset_period="daily"),
            QuotaType.STORAGE: QuotaLimit(QuotaType.STORAGE, 10 * 1024 * 1024 * 1024, reset_period="monthly"),  # 10GB
            QuotaType.FILE_UPLOADS: QuotaLimit(QuotaType.FILE_UPLOADS, 200, reset_period="daily"),
            QuotaType.TRAINING_JOBS: QuotaLimit(QuotaType.TRAINING_JOBS, 20, reset_period="monthly"),
        }
    
    @staticmethod
    def enterprise_tier() -> Dict[QuotaType, QuotaLimit]:
        """Enterprise tier quota limits."""
        return {
            QuotaType.TOKENS: QuotaLimit(QuotaType.TOKENS, 10000000, reset_period="monthly"),
            QuotaType.REQUESTS: QuotaLimit(QuotaType.REQUESTS, 100000, reset_period="daily"),
            QuotaType.STORAGE: QuotaLimit(QuotaType.STORAGE, 100 * 1024 * 1024 * 1024, reset_period="monthly"),  # 100GB
            QuotaType.FILE_UPLOADS: QuotaLimit(QuotaType.FILE_UPLOADS, 1000, reset_period="daily"),
            QuotaType.TRAINING_JOBS: QuotaLimit(QuotaType.TRAINING_JOBS, 100, reset_period="monthly"),
        }

class SessionRateLimitPresets:
    """Predefined rate limit configurations."""
    
    @staticmethod
    def default_limits() -> Dict[RateLimitType, RateLimit]:
        """Default rate limits."""
        return {
            RateLimitType.PER_SECOND: RateLimit(RateLimitType.PER_SECOND, 10, 1),  # 10 per second
            RateLimitType.PER_MINUTE: RateLimit(RateLimitType.PER_MINUTE, 100, 60),  # 100 per minute
            RateLimitType.PER_HOUR: RateLimit(RateLimitType.PER_HOUR, 1000, 3600),  # 1000 per hour
            RateLimitType.CONCURRENT: RateLimit(RateLimitType.CONCURRENT, 5, 1),  # 5 concurrent
        }
    
    @staticmethod
    def premium_limits() -> Dict[RateLimitType, RateLimit]:
        """Premium rate limits."""
        return {
            RateLimitType.PER_SECOND: RateLimit(RateLimitType.PER_SECOND, 50, 1),  # 50 per second
            RateLimitType.PER_MINUTE: RateLimit(RateLimitType.PER_MINUTE, 500, 60),  # 500 per minute
            RateLimitType.PER_HOUR: RateLimit(RateLimitType.PER_HOUR, 10000, 3600),  # 10000 per hour
            RateLimitType.CONCURRENT: RateLimit(RateLimitType.CONCURRENT, 20, 1),  # 20 concurrent
        }
