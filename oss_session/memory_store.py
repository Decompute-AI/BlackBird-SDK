"""MemoryStore for caching, vector search, and document embedding."""

import os
import json
import time
import threading
import pickle
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from collections import OrderedDict, defaultdict
import heapq

from .memory_types import (
    CacheEntry, VectorDocument, VectorSearchResult, CacheStats, VectorIndexStats,
    MemoryStoreConfig, EvictionPolicy, SimilarityMetric, EmbeddingModel,
    calculate_cosine_similarity, calculate_euclidean_distance, generate_cache_key
)
from oss_utils.errors import ValidationError, MemoryError
from oss_utils.feature_flags import require_feature, is_feature_enabled
from oss_utils.logger import get_logger

class MemoryStore:
    """Advanced memory store with caching and vector operations."""
    
    def __init__(self, config: MemoryStoreConfig = None):
        """Initialize the MemoryStore."""
        self.config = config or MemoryStoreConfig()
        self.logger = get_logger()
        
        # Cache storage
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = OrderedDict()  # For LRU
        self.access_frequency = defaultdict(int)  # For LFU
        
        # Vector storage
        self.vector_documents: Dict[str, VectorDocument] = {}
        self.vector_index = []  # Simple flat index
        
        # Statistics
        self.cache_stats = CacheStats()
        self.vector_stats = VectorIndexStats()
        
        # Embedding function
        self.embedding_function: Optional[Callable[[str], List[float]]] = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background cleanup
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        
        # Initialize embedding function
        self._initialize_embedding_function()
        
        # Start background cleanup
        self._start_cleanup_thread()
        
        self.logger.info("MemoryStore initialized", 
                        max_cache_size_mb=self.config.max_cache_size_mb,
                        max_cache_entries=self.config.max_cache_entries,
                        embedding_model=self.config.embedding_model.value)
    
    def _initialize_embedding_function(self):
        """Initialize embedding function based on configuration."""
        try:
            if self.config.embedding_model == EmbeddingModel.SENTENCE_TRANSFORMERS:
                try:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer('all-MiniLM-L6-v2')
                    self.embedding_function = lambda text: model.encode(text).tolist()
                    self.logger.info("Initialized SentenceTransformers embedding")
                except ImportError:
                    self.logger.warning("SentenceTransformers not available, using dummy embeddings")
                    self.embedding_function = self._dummy_embedding_function
            else:
                # For other models (OpenAI, etc.), use dummy for now
                self.logger.warning(f"Embedding model {self.config.embedding_model.value} not implemented, using dummy")
                self.embedding_function = self._dummy_embedding_function
                
        except Exception as e:
            self.logger.error("Failed to initialize embedding function", error=str(e))
            self.embedding_function = self._dummy_embedding_function
    
    def _dummy_embedding_function(self, text: str) -> List[float]:
        """Dummy embedding function for testing."""
        # Generate deterministic embeddings based on text hash
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to 384-dimensional vector (matching all-MiniLM-L6-v2)
        embedding = []
        for i in range(384):
            byte_idx = i % len(hash_bytes)
            embedding.append((hash_bytes[byte_idx] - 128) / 128.0)
        
        return embedding
    
    def set_cache(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Store data in cache with optional TTL."""
        if not key:
            raise ValidationError("Cache key cannot be empty", field_name="key")
        
        with self._lock:
            # Check if we need to evict entries
            self._evict_if_needed()
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                ttl=ttl or self.config.default_ttl
            )
            entry.calculate_size()
            
            # Store entry
            old_entry = self.cache.get(key)
            self.cache[key] = entry
            
            # Update access tracking
            self.access_order[key] = time.time()
            self.access_frequency[key] = 1
            
            # Update statistics
            if old_entry:
                self.cache_stats.total_size_bytes -= old_entry.size_bytes
            else:
                self.cache_stats.total_entries += 1
            
            self.cache_stats.total_size_bytes += entry.size_bytes
            
            self.logger.debug("Cache entry stored", 
                            key=key,
                            size_bytes=entry.size_bytes,
                            ttl=entry.ttl)
            
            return True
    
    def get_cache(self, key: str) -> Optional[Any]:
        """Retrieve data from cache."""
        if not key:
            return None
        
        with self._lock:
            entry = self.cache.get(key)
            
            if entry is None:
                self.cache_stats.miss_count += 1
                self.logger.debug("Cache miss", key=key)
                return None
            
            if entry.is_expired:
                self._remove_cache_entry(key)
                self.cache_stats.miss_count += 1
                self.cache_stats.expired_count += 1
                self.logger.debug("Cache entry expired", key=key)
                return None
            
            # Update access tracking
            entry.touch()
            self.access_order[key] = time.time()
            self.access_frequency[key] += 1
            
            self.cache_stats.hit_count += 1
            self.logger.debug("Cache hit", key=key, access_count=entry.access_count)
            
            return entry.value
    
    def delete_cache(self, key: str) -> bool:
        """Delete a cache entry."""
        with self._lock:
            if key in self.cache:
                self._remove_cache_entry(key)
                self.logger.debug("Cache entry deleted", key=key)
                return True
            return False
    
    def clear_cache(self) -> int:
        """Clear all cache entries."""
        with self._lock:
            count = len(self.cache)
            self.cache.clear()
            self.access_order.clear()
            self.access_frequency.clear()
            
            # Reset stats
            self.cache_stats = CacheStats()
            
            self.logger.info("Cache cleared", entries_removed=count)
            return count
    
    def embed_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None) -> VectorDocument:
        """Generate embedding for a document and store it."""
        if not doc_id or not content:
            raise ValidationError("Document ID and content are required")
        
        if not self.embedding_function:
            raise MemoryError("Embedding function not available")
        
        try:
            # Generate embedding
            embedding = self.embedding_function(content)
            
            if len(embedding) > self.config.max_vector_dimension:
                raise ValidationError(
                    f"Embedding dimension {len(embedding)} exceeds maximum {self.config.max_vector_dimension}"
                )
            
            # Create document
            document = VectorDocument(
                doc_id=doc_id,
                content=content,
                embedding=embedding,
                metadata=metadata or {}
            )
            
            with self._lock:
                # Store document
                old_doc = self.vector_documents.get(doc_id)
                self.vector_documents[doc_id] = document
                
                # Update index
                if old_doc:
                    # Remove old document from index
                    self.vector_index = [doc for doc in self.vector_index if doc.doc_id != doc_id]
                else:
                    self.vector_stats.total_documents += 1
                
                self.vector_index.append(document)
                
                # Update stats
                if not old_doc:
                    self.vector_stats.embedding_dimension = len(embedding)
                
                self.logger.debug("Document embedded", 
                                doc_id=doc_id,
                                content_length=len(content),
                                embedding_dimension=len(embedding))
                
                return document
                
        except Exception as e:
            self.logger.error("Failed to embed document", doc_id=doc_id, error=str(e))
            raise MemoryError(f"Document embedding failed: {str(e)}")
    
    def vector_search(self, query: str, limit: int = 10, threshold: float = 0.0) -> List[VectorSearchResult]:
        """Perform vector similarity search."""
        if not query:
            raise ValidationError("Query cannot be empty", field_name="query")
        
        if not self.embedding_function:
            raise MemoryError("Embedding function not available")
        
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_function(query)
            
            with self._lock:
                if not self.vector_index:
                    return []
                
                # Calculate similarities
                similarities = []
                for doc in self.vector_index:
                    if self.config.similarity_metric == SimilarityMetric.COSINE:
                        similarity = calculate_cosine_similarity(query_embedding, doc.embedding)
                        distance = 1.0 - similarity
                    elif self.config.similarity_metric == SimilarityMetric.EUCLIDEAN:
                        distance = calculate_euclidean_distance(query_embedding, doc.embedding)
                        similarity = 1.0 / (1.0 + distance)
                    else:
                        # Default to cosine
                        similarity = calculate_cosine_similarity(query_embedding, doc.embedding)
                        distance = 1.0 - similarity
                    
                    if similarity >= threshold:
                        similarities.append((doc, similarity, distance))
                
                # Sort by similarity (descending)
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # Limit results
                similarities = similarities[:limit]
                
                # Create result objects
                results = []
                for rank, (doc, similarity, distance) in enumerate(similarities):
                    result = VectorSearchResult(
                        document=doc,
                        similarity_score=similarity,
                        distance=distance,
                        rank=rank + 1
                    )
                    results.append(result)
                
                # Update stats
                search_time_ms = (time.time() - start_time) * 1000
                self.vector_stats.search_count += 1
                self.vector_stats.average_search_time_ms = (
                    (self.vector_stats.average_search_time_ms * (self.vector_stats.search_count - 1) + search_time_ms) 
                    / self.vector_stats.search_count
                )
                
                self.logger.debug("Vector search completed", 
                                query_length=len(query),
                                results_count=len(results),
                                search_time_ms=search_time_ms)
                
                return results
                
        except Exception as e:
            self.logger.error("Vector search failed", query=query[:100], error=str(e))
            raise MemoryError(f"Vector search failed: {str(e)}")
    
    def index_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Index a document for vector search."""
        try:
            self.embed_document(doc_id, content, metadata)
            return True
        except Exception as e:
            self.logger.error("Failed to index document", doc_id=doc_id, error=str(e))
            return False
    
    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Get a document by ID."""
        with self._lock:
            return self.vector_documents.get(doc_id)
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the vector store."""
        with self._lock:
            if doc_id in self.vector_documents:
                del self.vector_documents[doc_id]
                self.vector_index = [doc for doc in self.vector_index if doc.doc_id != doc_id]
                self.vector_stats.total_documents -= 1
                
                self.logger.debug("Document deleted", doc_id=doc_id)
                return True
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            stats_dict = {
                'total_entries': self.cache_stats.total_entries,
                'total_size_mb': self.cache_stats.size_mb,
                'hit_count': self.cache_stats.hit_count,
                'miss_count': self.cache_stats.miss_count,
                'hit_rate': self.cache_stats.hit_rate,
                'eviction_count': self.cache_stats.eviction_count,
                'expired_count': self.cache_stats.expired_count,
                'config': self.config.to_dict()
            }
            return stats_dict
    
    def get_vector_stats(self) -> Dict[str, Any]:
        """Get vector index statistics."""
        with self._lock:
            stats_dict = {
                'total_documents': self.vector_stats.total_documents,
                'index_size_mb': self.vector_stats.size_mb,
                'embedding_dimension': self.vector_stats.embedding_dimension,
                'search_count': self.vector_stats.search_count,
                'average_search_time_ms': self.vector_stats.average_search_time_ms,
                'similarity_metric': self.config.similarity_metric.value,
                'embedding_model': self.config.embedding_model.value
            }
            return stats_dict
    
    def optimize_index(self) -> bool:
        """Optimize the vector index for better performance."""
        with self._lock:
            # For now, just rebuild the index
            # In a real implementation, this could build more sophisticated indexes
            self.vector_index = list(self.vector_documents.values())
            
            self.logger.info("Vector index optimized", 
                           documents=len(self.vector_index))
            return True
    
    def _evict_if_needed(self):
        """Evict cache entries if limits are exceeded."""
        # Check size limit
        while (self.cache_stats.total_size_bytes > self.config.max_cache_size_mb * 1024 * 1024 or
               len(self.cache) >= self.config.max_cache_entries):
            
            if not self.cache:
                break
                
            if self.config.eviction_policy == EvictionPolicy.LRU:
                # Remove least recently used
                oldest_key = min(self.access_order.keys(), key=self.access_order.get)
                self._remove_cache_entry(oldest_key)
            elif self.config.eviction_policy == EvictionPolicy.LFU:
                # Remove least frequently used
                least_used_key = min(self.access_frequency.keys(), key=self.access_frequency.get)
                self._remove_cache_entry(least_used_key)
            elif self.config.eviction_policy == EvictionPolicy.FIFO:
                # Remove oldest entry
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
                self._remove_cache_entry(oldest_key)
            else:
                # Default to LRU
                oldest_key = min(self.access_order.keys(), key=self.access_order.get)
                self._remove_cache_entry(oldest_key)
            
            self.cache_stats.eviction_count += 1
    
    def _remove_cache_entry(self, key: str):
        """Remove a cache entry and update tracking."""
        if key in self.cache:
            entry = self.cache[key]
            del self.cache[key]
            
            if key in self.access_order:
                del self.access_order[key]
            if key in self.access_frequency:
                del self.access_frequency[key]
            
            self.cache_stats.total_entries -= 1
            self.cache_stats.total_size_bytes -= entry.size_bytes
    
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
                self._cleanup_expired_entries()
                self._stop_cleanup.wait(self.config.cleanup_interval)
            except Exception as e:
                self.logger.error("Error in cleanup loop", error=str(e))
                time.sleep(5)
    
    def _cleanup_expired_entries(self):
        """Clean up expired cache entries."""
        expired_keys = []
        
        with self._lock:
            for key, entry in self.cache.items():
                if entry.is_expired:
                    expired_keys.append(key)
        
        for key in expired_keys:
            with self._lock:
                self._remove_cache_entry(key)
                self.cache_stats.expired_count += 1
        
        if expired_keys:
            self.logger.debug("Cleaned up expired entries", count=len(expired_keys))
    
    def cleanup(self):
        """Clean up resources."""
        self.logger.info("Cleaning up MemoryStore")
        
        # Stop cleanup thread
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        
        # Clear all data
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.access_frequency.clear()
            self.vector_documents.clear()
            self.vector_index.clear()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass
