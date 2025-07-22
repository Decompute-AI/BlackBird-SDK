"""Memory store types and configurations for caching and vector operations."""

import time
import hashlib
import pickle
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
# Add this to the existing memory_types.py file

# Create alias for backward compatibility
# MemoryConfig = MemoryStoreConfig

@dataclass
class SessionMemoryConfig:
    """Memory configuration for sessions."""
    enable_memory: bool = True
    memory_type: str = "vector"
    max_memory_size_mb: int = 50
    memory_ttl: Optional[float] = 3600
    enable_conversation_history: bool = True
    max_conversation_length: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'enable_memory': self.enable_memory,
            'memory_type': self.memory_type,
            'max_memory_size_mb': self.max_memory_size_mb,
            'memory_ttl': self.memory_ttl,
            'enable_conversation_history': self.enable_conversation_history,
            'max_conversation_length': self.max_conversation_length
        }

class CacheType(Enum):
    """Types of cache storage."""
    MEMORY = "memory"
    PERSISTENT = "persistent"
    DISTRIBUTED = "distributed"

class VectorIndexType(Enum):
    """Types of vector indexes."""
    FLAT = "flat"
    IVF = "ivf"
    HNSW = "hnsw"
    LSH = "lsh"

class EmbeddingModel(Enum):
    """Supported embedding models."""
    SENTENCE_TRANSFORMERS = "sentence-transformers/all-MiniLM-L6-v2"
    OPENAI_ADA = "text-embedding-ada-002"
    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_3_LARGE = "text-embedding-3-large"
    HUGGINGFACE_BGE = "BAAI/bge-small-en-v1.5"

@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    @property
    def age_seconds(self) -> float:
        """Get entry age in seconds."""
        return time.time() - self.created_at
    
    @property
    def idle_seconds(self) -> float:
        """Get idle time since last access."""
        return time.time() - self.last_accessed
    
    def touch(self):
        """Update last accessed time and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def calculate_size(self):
        """Calculate and update size in bytes."""
        try:
            self.size_bytes = len(pickle.dumps(self.value))
        except:
            self.size_bytes = 0

@dataclass
class VectorDocument:
    """Represents a document with vector embedding."""
    doc_id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return len(self.embedding) if self.embedding else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'doc_id': self.doc_id,
            'content': self.content,
            'embedding': self.embedding,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'embedding_dimension': self.embedding_dimension
        }

@dataclass
class VectorSearchResult:
    """Represents a vector search result."""
    document: VectorDocument
    similarity_score: float
    distance: float
    rank: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'document': self.document.to_dict(),
            'similarity_score': self.similarity_score,
            'distance': self.distance,
            'rank': self.rank
        }

@dataclass
class CacheStats:
    """Cache statistics."""
    total_entries: int = 0
    total_size_bytes: int = 0
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    expired_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.hit_count + self.miss_count
        return self.hit_count / total_requests if total_requests > 0 else 0.0
    
    @property
    def size_mb(self) -> float:
        """Get total size in megabytes."""
        return self.total_size_bytes / (1024 * 1024)

@dataclass
class VectorIndexStats:
    """Vector index statistics."""
    total_documents: int = 0
    index_size_bytes: int = 0
    embedding_dimension: int = 0
    index_type: str = "flat"
    search_count: int = 0
    average_search_time_ms: float = 0.0
    
    @property
    def size_mb(self) -> float:
        """Get index size in megabytes."""
        return self.index_size_bytes / (1024 * 1024)

class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live based
    SIZE = "size"  # Size based

class SimilarityMetric(Enum):
    """Similarity metrics for vector search."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"

@dataclass
class MemoryStoreConfig:
    """Configuration for MemoryStore."""
    max_cache_size_mb: int = 100
    max_cache_entries: int = 10000
    default_ttl: Optional[float] = 3600  # 1 hour
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    cleanup_interval: int = 300  # 5 minutes
    
    # Vector configuration
    embedding_model: EmbeddingModel = EmbeddingModel.SENTENCE_TRANSFORMERS
    vector_index_type: VectorIndexType = VectorIndexType.FLAT
    similarity_metric: SimilarityMetric = SimilarityMetric.COSINE
    max_vector_dimension: int = 1536
    
    # Performance settings
    enable_persistence: bool = False
    persistence_path: Optional[str] = None
    batch_size: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'max_cache_size_mb': self.max_cache_size_mb,
            'max_cache_entries': self.max_cache_entries,
            'default_ttl': self.default_ttl,
            'eviction_policy': self.eviction_policy.value,
            'cleanup_interval': self.cleanup_interval,
            'embedding_model': self.embedding_model.value,
            'vector_index_type': self.vector_index_type.value,
            'similarity_metric': self.similarity_metric.value,
            'max_vector_dimension': self.max_vector_dimension,
            'enable_persistence': self.enable_persistence,
            'persistence_path': self.persistence_path,
            'batch_size': self.batch_size
        }

def calculate_cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimension")
    
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(x * x for x in b) ** 0.5
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    return dot_product / (magnitude_a * magnitude_b)

def calculate_euclidean_distance(a: List[float], b: List[float]) -> float:
    """Calculate Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimension")
    
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

def generate_cache_key(*args, **kwargs) -> str:
    """Generate a cache key from arguments."""
    key_data = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_data.encode()).hexdigest()
