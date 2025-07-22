"""
Graph-based Memory System for Long-term AI Memory Management
Supports facts, relationships, episodic memory, and semantic memory with both Neo4j and NetworkX backends.
"""

import os
import json
import uuid
import hashlib
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Graph database imports
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: NetworkX not available. Using fallback graph implementation.")

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("Warning: Neo4j not available. Using NetworkX backend.")

# Vector similarity imports
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    VECTOR_AVAILABLE = True
except ImportError:
    VECTOR_AVAILABLE = False
    print("Warning: Vector similarity not available. Using basic similarity.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory nodes"""
    FACT = "fact"
    CONCEPT = "concept"
    CONVERSATION = "conversation"
    USER = "user"
    AGENT = "agent"
    DOCUMENT = "document"
    RELATIONSHIP = "relationship"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class RelationshipType(Enum):
    """Types of relationships between memory nodes"""
    SIMILAR_TO = "similar_to"
    CONTAINS = "contains"
    DISCUSSED = "discussed"
    REMEMBERS = "remembers"
    EVOLVED_FROM = "evolved_from"
    ACCESSED_BY = "accessed_by"
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    CAUSES = "causes"
    IMPLIES = "implies"


@dataclass
class MemoryNode:
    """Represents a memory node in the graph"""
    id: str
    type: MemoryType
    content: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    confidence: float = 1.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['type'] = self.type.value  # Convert enum to string
        if self.last_accessed:
            data['last_accessed'] = self.last_accessed.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryNode':
        """Create from dictionary"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        data['type'] = MemoryType(data['type'])  # Convert string back to enum
        if data.get('last_accessed'):
            data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        return cls(**data)


@dataclass
class MemoryRelationship:
    """Represents a relationship between memory nodes"""
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    metadata: Dict[str, Any]
    created_at: datetime
    strength: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['relationship_type'] = self.relationship_type.value  # Convert enum to string
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryRelationship':
        """Create from dictionary"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['relationship_type'] = RelationshipType(data['relationship_type'])  # Convert string back to enum
        return cls(**data)


class GraphMemoryManager:
    """
    Graph-based memory management system supporting long-term AI memory
    with facts, relationships, and semantic similarity.
    """
    
    def __init__(self, 
                 backend: str = "networkx",
                 db_path: str = "graph_memory.db",
                 neo4j_uri: Optional[str] = None,
                 neo4j_user: Optional[str] = None,
                 neo4j_password: Optional[str] = None):
        """
        Initialize the graph memory manager.
        
        Args:
            backend: "networkx" or "neo4j"
            db_path: Path for local storage
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        self.backend = backend
        self.db_path = db_path
        self.lock = threading.RLock()
        
        # Initialize backend
        if backend == "neo4j" and NEO4J_AVAILABLE:
            self._init_neo4j(neo4j_uri, neo4j_user, neo4j_password)
        else:
            self._init_networkx()
        
        # Initialize vector similarity if available
        if VECTOR_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.vector_dimension = self.embedding_model.get_sentence_embedding_dimension()
            except Exception as e:
                logger.warning(f"Could not initialize embedding model: {e}")
                self.embedding_model = None
                self.vector_dimension = None
        else:
            self.embedding_model = None
            self.vector_dimension = None
        
        # Memory statistics (only initialize if not already done in _init_networkx)
        if not hasattr(self, 'stats'):
            self.stats = {
                'total_nodes': 0,
                'total_relationships': 0,
                'memory_types': {},
                'last_backup': None
            }
        
        logger.info(f"GraphMemoryManager initialized with {backend} backend")
    
    def _init_networkx(self):
        """Initialize NetworkX backend"""
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX is required for networkx backend")
        
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, MemoryNode] = {}
        self.relationships: Dict[str, MemoryRelationship] = {}
        
        self._load_from_disk()
    
    def _init_neo4j(self, uri: str, user: str, password: str):
        """Initialize Neo4j backend"""
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver is required for neo4j backend")
        
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._create_neo4j_constraints()
    
    def _create_neo4j_constraints(self):
        """Create Neo4j constraints and indexes"""
        with self.driver.session() as session:
            # Create constraints
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:MemoryNode) REQUIRE n.id IS UNIQUE")
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:MemoryNode) ON (n.type)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:MemoryNode) ON (n.content)")
    
    def _load_from_disk(self):
        """Load memory from disk (NetworkX backend only)"""
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load nodes
                for node_data in data.get('nodes', []):
                    node = MemoryNode.from_dict(node_data)
                    self.nodes[node.id] = node
                    self.graph.add_node(node.id, 
                                      id=node.id,
                                      type=node.type.value,
                                      content=node.content,
                                      metadata=node.metadata,
                                      created_at=node.created_at.isoformat(),
                                      updated_at=node.updated_at.isoformat(),
                                      confidence=node.confidence,
                                      access_count=node.access_count)
                
                # Load relationships
                for rel_data in data.get('relationships', []):
                    rel = MemoryRelationship.from_dict(rel_data)
                    self.relationships[f"{rel.source_id}_{rel.target_id}_{rel.relationship_type.value}"] = rel
                    self.graph.add_edge(rel.source_id, rel.target_id, 
                                      relationship_type=rel.relationship_type.value,
                                      source_id=rel.source_id,
                                      target_id=rel.target_id,
                                      strength=rel.strength,
                                      created_at=rel.created_at.isoformat(),
                                      metadata=rel.metadata)
                
                # Load stats (ensure stats attribute exists)
                if not hasattr(self, 'stats'):
                    self.stats = {
                        'total_nodes': 0,
                        'total_relationships': 0,
                        'memory_types': {},
                        'last_backup': None
                    }
                
                self.stats = data.get('stats', self.stats)
                logger.info(f"Loaded {len(self.nodes)} nodes and {len(self.relationships)} relationships")
        except Exception as e:
            logger.error(f"Error loading memory from disk: {e}")
            # Ensure stats are initialized even if loading fails
            if not hasattr(self, 'stats'):
                self.stats = {
                    'total_nodes': 0,
                    'total_relationships': 0,
                    'memory_types': {},
                    'last_backup': None
                }
    
    def _save_to_disk(self):
        """Save memory to disk (NetworkX backend only)"""
        try:
            # Update stats before saving
            self._update_stats()
            
            data = {
                'nodes': [node.to_dict() for node in self.nodes.values()],
                'relationships': [rel.to_dict() for rel in self.relationships.values()],
                'stats': self.stats
            }
            
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.stats['last_backup'] = datetime.now().isoformat()
            logger.info("Memory saved to disk")
        except Exception as e:
            logger.error(f"Error saving memory to disk: {e}")
    
    def add_fact(self, 
                 content: str, 
                 fact_type: str = "general",
                 metadata: Optional[Dict[str, Any]] = None,
                 confidence: float = 1.0) -> str:
        """
        Add a new fact to memory.
        
        Args:
            content: The fact content
            fact_type: Type of fact (e.g., "general", "technical", "personal")
            metadata: Additional metadata
            confidence: Confidence level (0.0 to 1.0)
            
        Returns:
            str: The fact ID
        """
        with self.lock:
            # Check for duplicates
            duplicate_id = self._find_duplicate_fact(content)
            if duplicate_id:
                logger.info(f"Fact already exists with ID: {duplicate_id}")
                return duplicate_id
            
            # Create new fact
            fact_id = str(uuid.uuid4())
            now = datetime.now()
            
            node = MemoryNode(
                id=fact_id,
                type=MemoryType.FACT,
                content=content,
                metadata={
                    'fact_type': fact_type,
                    'content_hash': self._hash_content(content),
                    **(metadata or {})
                },
                created_at=now,
                updated_at=now,
                confidence=confidence
            )
            
            if self.backend == "networkx":
                self.nodes[fact_id] = node
                self.graph.add_node(fact_id, 
                                  id=node.id,
                                  type=node.type.value,
                                  content=node.content,
                                  metadata=node.metadata,
                                  created_at=node.created_at.isoformat(),
                                  updated_at=node.updated_at.isoformat(),
                                  confidence=node.confidence,
                                  access_count=node.access_count)
                self._save_to_disk()
            else:
                self._add_node_to_neo4j(node)
            
            self.stats['total_nodes'] += 1
            self.stats['memory_types'][MemoryType.FACT.value] = \
                self.stats['memory_types'].get(MemoryType.FACT.value, 0) + 1
            
            logger.info(f"Added fact: {content[:50]}...")
            return fact_id
    
    def _find_duplicate_fact(self, content: str) -> Optional[str]:
        """Find duplicate facts based on content similarity"""
        content_hash = self._hash_content(content)
        
        if self.backend == "networkx":
            for node in self.nodes.values():
                if (node.type == MemoryType.FACT and 
                    node.metadata.get('content_hash') == content_hash):
                    return node.id
        else:
            with self.driver.session() as session:
                result = session.run(
                    "MATCH (n:MemoryNode) WHERE n.type = 'fact' AND n.metadata.content_hash = $hash RETURN n.id LIMIT 1",
                    hash=content_hash
                )
                record = result.single()
                if record:
                    return record["n.id"]
        
        return None
    
    def _hash_content(self, content: str) -> str:
        """Create hash of content for duplicate detection"""
        return hashlib.md5(content.lower().strip().encode()).hexdigest()
    
    def get_fact(self, fact_id: str) -> Optional[MemoryNode]:
        """Get a fact by ID"""
        with self.lock:
            if self.backend == "networkx":
                return self.nodes.get(fact_id)
            else:
                return self._get_node_from_neo4j(fact_id)
    
    def get_all_facts(self, 
                     fact_type: Optional[str] = None,
                     limit: int = 100,
                     offset: int = 0) -> List[MemoryNode]:
        """Get all facts with optional filtering"""
        with self.lock:
            if self.backend == "networkx":
                facts = []
                for node in self.nodes.values():
                    if node.type == MemoryType.FACT:
                        if fact_type is None or node.metadata.get('fact_type') == fact_type:
                            facts.append(node)
                
                # Sort by creation date (newest first)
                facts.sort(key=lambda x: x.created_at, reverse=True)
                return facts[offset:offset + limit]
            else:
                return self._get_facts_from_neo4j(fact_type, limit, offset)
    
    def update_fact(self, 
                   fact_id: str, 
                   new_content: str = None,
                   new_metadata: Optional[Dict[str, Any]] = None,
                   new_confidence: Optional[float] = None) -> bool:
        """Update an existing fact"""
        with self.lock:
            if self.backend == "networkx":
                if fact_id not in self.nodes:
                    return False
                
                node = self.nodes[fact_id]
                if new_content:
                    node.content = new_content
                    node.metadata['content_hash'] = self._hash_content(new_content)
                if new_metadata:
                    node.metadata.update(new_metadata)
                if new_confidence is not None:
                    node.confidence = new_confidence
                
                node.updated_at = datetime.now()
                self.graph.nodes[fact_id].update({
                    'content': node.content,
                    'metadata': node.metadata,
                    'confidence': node.confidence,
                    'updated_at': node.updated_at.isoformat()
                })
                self._save_to_disk()
                return True
            else:
                return self._update_node_in_neo4j(fact_id, new_content, new_metadata, new_confidence)
    
    def delete_fact(self, fact_id: str) -> bool:
        """Delete a fact and its relationships"""
        with self.lock:
            if self.backend == "networkx":
                if fact_id not in self.nodes:
                    return False
                
                # Remove relationships
                edges_to_remove = []
                for source, target, key, data in self.graph.edges(data=True, keys=True):
                    if source == fact_id or target == fact_id:
                        edges_to_remove.append((source, target, key))
                
                for source, target, key in edges_to_remove:
                    self.graph.remove_edge(source, target, key)
                
                # Remove node
                del self.nodes[fact_id]
                self.graph.remove_node(fact_id)
                
                self.stats['total_nodes'] -= 1
                self.stats['memory_types'][MemoryType.FACT.value] -= 1
                self._save_to_disk()
                return True
            else:
                return self._delete_node_from_neo4j(fact_id)
    
    def add_relationship(self, 
                        source_id: str, 
                        target_id: str, 
                        relationship_type: RelationshipType,
                        metadata: Optional[Dict[str, Any]] = None,
                        strength: float = 1.0) -> bool:
        """Add a relationship between two memory nodes"""
        with self.lock:
            # Check if both nodes exist
            if not (self.get_fact(source_id) and self.get_fact(target_id)):
                return False
            
            rel_key = f"{source_id}_{target_id}_{relationship_type.value}"
            
            if self.backend == "networkx":
                if rel_key in self.relationships:
                    return False  # Relationship already exists
                
                rel = MemoryRelationship(
                    source_id=source_id,
                    target_id=target_id,
                    relationship_type=relationship_type,
                    metadata=metadata or {},
                    created_at=datetime.now(),
                    strength=strength
                )
                
                self.relationships[rel_key] = rel
                self.graph.add_edge(source_id, target_id, 
                                  relationship_type=relationship_type.value,
                                  source_id=rel.source_id,
                                  target_id=rel.target_id,
                                  strength=rel.strength,
                                  created_at=rel.created_at.isoformat(),
                                  metadata=rel.metadata)
                self._save_to_disk()
            else:
                return self._add_relationship_to_neo4j(source_id, target_id, relationship_type, metadata, strength)
            
            self.stats['total_relationships'] += 1
            return True
    
    def get_related_facts(self, 
                         fact_id: str, 
                         relationship_type: Optional[RelationshipType] = None,
                         limit: int = 10) -> List[Tuple[MemoryNode, RelationshipType, float]]:
        """Get facts related to a given fact"""
        with self.lock:
            if self.backend == "networkx":
                related = []
                for source, target, key, data in self.graph.edges(data=True, keys=True):
                    if source == fact_id:
                        if relationship_type is None or data.get('relationship_type') == relationship_type.value:
                            target_node = self.nodes.get(target)
                            if target_node:
                                rel_type = RelationshipType(data.get('relationship_type', 'related_to'))
                                strength = data.get('strength', 1.0)
                                related.append((target_node, rel_type, strength))
                
                # Sort by strength and limit
                related.sort(key=lambda x: x[2], reverse=True)
                return related[:limit]
            else:
                return self._get_related_facts_from_neo4j(fact_id, relationship_type, limit)
    
    def search_facts(self, 
                    query: str, 
                    fact_type: Optional[str] = None,
                    limit: int = 10,
                    similarity_threshold: float = 0.7) -> List[Tuple[MemoryNode, float]]:
        """Search facts using semantic similarity"""
        with self.lock:
            if not self.embedding_model:
                # Fallback to basic text search
                return self._basic_text_search(query, fact_type, limit)
            
            # Get query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            if self.backend == "networkx":
                results = []
                for node in self.nodes.values():
                    if node.type == MemoryType.FACT:
                        if fact_type and node.metadata.get('fact_type') != fact_type:
                            continue
                        
                        # Get node embedding (cache or compute)
                        node_embedding = self._get_node_embedding(node)
                        if node_embedding is not None:
                            similarity = self._cosine_similarity(query_embedding, node_embedding)
                            if similarity >= similarity_threshold:
                                results.append((node, similarity))
                
                # Sort by similarity
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:limit]
            else:
                return self._search_facts_in_neo4j(query, fact_type, limit, similarity_threshold)
    
    def _basic_text_search(self, 
                          query: str, 
                          fact_type: Optional[str] = None,
                          limit: int = 10) -> List[Tuple[MemoryNode, float]]:
        """Basic text search fallback"""
        query_words = set(query.lower().split())
        results = []
        
        for node in self.nodes.values():
            if node.type == MemoryType.FACT:
                if fact_type and node.metadata.get('fact_type') != fact_type:
                    continue
                
                content_words = set(node.content.lower().split())
                overlap = len(query_words.intersection(content_words))
                if overlap > 0:
                    similarity = overlap / len(query_words)
                    results.append((node, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def _get_node_embedding(self, node: MemoryNode) -> Optional[np.ndarray]:
        """Get or compute embedding for a node"""
        if not self.embedding_model:
            return None
        
        # Check if embedding is cached
        if 'embedding' not in node.metadata:
            try:
                embedding = self.embedding_model.encode([node.content])[0]
                node.metadata['embedding'] = embedding.tolist()
                return embedding
            except Exception as e:
                logger.error(f"Error computing embedding: {e}")
                return None
        
        return np.array(node.metadata['embedding'])
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return dot_product / (norm1 * norm2)
        except Exception:
            return 0.0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        with self.lock:
            stats = self.stats.copy()
            
            if self.backend == "networkx":
                stats['total_nodes'] = len(self.nodes)
                # Count actual unique relationships from the relationships dict
                stats['total_relationships'] = len(self.relationships)
                
                # Count by type
                type_counts = {}
                for node in self.nodes.values():
                    node_type = node.type.value
                    type_counts[node_type] = type_counts.get(node_type, 0) + 1
                stats['memory_types'] = type_counts
                
                # Recent activity
                now = datetime.now()
                recent_nodes = sum(1 for node in self.nodes.values() 
                                 if (now - node.created_at).days <= 7)
                stats['recent_activity'] = recent_nodes
            
            return stats
    
    def clear_memory(self, memory_type: Optional[MemoryType] = None) -> int:
        """Clear all memory or specific type"""
        with self.lock:
            if self.backend == "networkx":
                if memory_type is None:
                    # Clear all
                    count = len(self.nodes)
                    self.nodes.clear()
                    self.relationships.clear()
                    self.graph.clear()
                    self.stats = {
                        'total_nodes': 0,
                        'total_relationships': 0,
                        'memory_types': {},
                        'last_backup': None
                    }
                else:
                    # Clear specific type
                    count = 0
                    nodes_to_remove = []
                    for node_id, node in self.nodes.items():
                        if node.type == memory_type:
                            nodes_to_remove.append(node_id)
                            count += 1
                    
                    for node_id in nodes_to_remove:
                        self.delete_fact(node_id)
                
                self._save_to_disk()
                return count
            else:
                return self._clear_memory_in_neo4j(memory_type)
    
    def backup_memory(self, backup_path: str) -> bool:
        """Backup memory to file"""
        try:
            with self.lock:
                if self.backend == "networkx":
                    data = {
                        'nodes': [node.to_dict() for node in self.nodes.values()],
                        'relationships': [rel.to_dict() for rel in self.relationships.values()],
                        'stats': self.stats,
                        'backup_timestamp': datetime.now().isoformat()
                    }
                    
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    
                    self.stats['last_backup'] = datetime.now().isoformat()
                    return True
                else:
                    return self._backup_neo4j_memory(backup_path)
        except Exception as e:
            logger.error(f"Error backing up memory: {e}")
            return False
    
    # Neo4j helper methods
    def _add_node_to_neo4j(self, node: MemoryNode):
        """Add node to Neo4j"""
        with self.driver.session() as session:
            session.run("""
                CREATE (n:MemoryNode {
                    id: $id, type: $type, content: $content, 
                    metadata: $metadata, created_at: $created_at,
                    updated_at: $updated_at, confidence: $confidence,
                    access_count: $access_count
                })
            """, **node.to_dict())
    
    def _get_node_from_neo4j(self, node_id: str) -> Optional[MemoryNode]:
        """Get node from Neo4j"""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (n:MemoryNode {id: $id}) RETURN n",
                id=node_id
            )
            record = result.single()
            if record:
                data = record["n"]
                return MemoryNode.from_dict(data)
        return None
    
    def _get_facts_from_neo4j(self, fact_type: Optional[str], limit: int, offset: int) -> List[MemoryNode]:
        """Get facts from Neo4j"""
        with self.driver.session() as session:
            if fact_type:
                result = session.run("""
                    MATCH (n:MemoryNode)
                    WHERE n.type = 'fact' AND n.metadata.fact_type = $fact_type
                    RETURN n ORDER BY n.created_at DESC
                    SKIP $offset LIMIT $limit
                """, fact_type=fact_type, offset=offset, limit=limit)
            else:
                result = session.run("""
                    MATCH (n:MemoryNode)
                    WHERE n.type = 'fact'
                    RETURN n ORDER BY n.created_at DESC
                    SKIP $offset LIMIT $limit
                """, offset=offset, limit=limit)
            
            return [MemoryNode.from_dict(record["n"]) for record in result]
    
    def _update_node_in_neo4j(self, node_id: str, new_content: str, new_metadata: Dict, new_confidence: float) -> bool:
        """Update node in Neo4j"""
        with self.driver.session() as session:
            updates = []
            params = {"id": node_id}
            
            if new_content:
                updates.append("n.content = $content")
                params["content"] = new_content
            if new_metadata:
                updates.append("n.metadata = $metadata")
                params["metadata"] = new_metadata
            if new_confidence is not None:
                updates.append("n.confidence = $confidence")
                params["confidence"] = new_confidence
            
            if updates:
                updates.append("n.updated_at = $updated_at")
                params["updated_at"] = datetime.now().isoformat()
                
                query = f"MATCH (n:MemoryNode {{id: $id}}) SET {', '.join(updates)}"
                result = session.run(query, **params)
                return result.consume().counters.nodes_updated > 0
        
        return False
    
    def _delete_node_from_neo4j(self, node_id: str) -> bool:
        """Delete node from Neo4j"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:MemoryNode {id: $id})
                OPTIONAL MATCH (n)-[r]-()
                DELETE r, n
            """, id=node_id)
            return result.consume().counters.nodes_deleted > 0
    
    def _add_relationship_to_neo4j(self, source_id: str, target_id: str, rel_type: RelationshipType, metadata: Dict, strength: float) -> bool:
        """Add relationship to Neo4j"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (source:MemoryNode {id: $source_id})
                MATCH (target:MemoryNode {id: $target_id})
                CREATE (source)-[r:RELATES_TO {
                    type: $rel_type, metadata: $metadata, 
                    strength: $strength, created_at: $created_at
                }]->(target)
            """, source_id=source_id, target_id=target_id, 
                 rel_type=rel_type.value, metadata=metadata or {},
                 strength=strength, created_at=datetime.now().isoformat())
            return result.consume().counters.relationships_created > 0
    
    def _get_related_facts_from_neo4j(self, fact_id: str, rel_type: Optional[RelationshipType], limit: int) -> List[Tuple[MemoryNode, RelationshipType, float]]:
        """Get related facts from Neo4j"""
        with self.driver.session() as session:
            if rel_type:
                result = session.run("""
                    MATCH (source:MemoryNode {id: $fact_id})-[r:RELATES_TO]->(target:MemoryNode)
                    WHERE r.type = $rel_type
                    RETURN target, r.type, r.strength
                    ORDER BY r.strength DESC LIMIT $limit
                """, fact_id=fact_id, rel_type=rel_type.value, limit=limit)
            else:
                result = session.run("""
                    MATCH (source:MemoryNode {id: $fact_id})-[r:RELATES_TO]->(target:MemoryNode)
                    RETURN target, r.type, r.strength
                    ORDER BY r.strength DESC LIMIT $limit
                """, fact_id=fact_id, limit=limit)
            
            return [(MemoryNode.from_dict(record["target"]), 
                    RelationshipType(record["r.type"]), 
                    record["r.strength"]) for record in result]
    
    def _search_facts_in_neo4j(self, query: str, fact_type: Optional[str], limit: int, threshold: float) -> List[Tuple[MemoryNode, float]]:
        """Search facts in Neo4j (simplified - would need vector search extension)"""
        # This is a simplified implementation
        # In production, you'd use Neo4j's vector search capabilities
        return self._basic_text_search(query, fact_type, limit)
    
    def _clear_memory_in_neo4j(self, memory_type: Optional[MemoryType]) -> int:
        """Clear memory in Neo4j"""
        with self.driver.session() as session:
            if memory_type:
                result = session.run("""
                    MATCH (n:MemoryNode {type: $type})
                    OPTIONAL MATCH (n)-[r]-()
                    DELETE r, n
                """, type=memory_type.value)
            else:
                result = session.run("""
                    MATCH (n:MemoryNode)
                    OPTIONAL MATCH (n)-[r]-()
                    DELETE r, n
                """)
            return result.consume().counters.nodes_deleted
    
    def _backup_neo4j_memory(self, backup_path: str) -> bool:
        """Backup Neo4j memory (simplified)"""
        # In production, you'd use Neo4j's backup tools
        return True

    def get_facts_by_type(self, fact_type: str, limit: int = 100) -> List[MemoryNode]:
        """Get facts by specific type"""
        return self.get_all_facts(fact_type=fact_type, limit=limit)
    
    def get_recent_facts(self, days: int = 7, limit: int = 50) -> List[MemoryNode]:
        """Get facts created in the last N days"""
        with self.lock:
            if self.backend == "networkx":
                cutoff_date = datetime.now() - timedelta(days=days)
                recent_facts = []
                
                for node in self.nodes.values():
                    if node.type == MemoryType.FACT and node.created_at >= cutoff_date:
                        recent_facts.append(node)
                
                recent_facts.sort(key=lambda x: x.created_at, reverse=True)
                return recent_facts[:limit]
            else:
                return self._get_recent_facts_from_neo4j(days, limit)
    
    def get_facts_by_confidence(self, min_confidence: float = 0.5, limit: int = 100) -> List[MemoryNode]:
        """Get facts with confidence above threshold"""
        with self.lock:
            if self.backend == "networkx":
                high_confidence_facts = []
                
                for node in self.nodes.values():
                    if node.type == MemoryType.FACT and node.confidence >= min_confidence:
                        high_confidence_facts.append(node)
                
                high_confidence_facts.sort(key=lambda x: x.confidence, reverse=True)
                return high_confidence_facts[:limit]
            else:
                return self._get_facts_by_confidence_from_neo4j(min_confidence, limit)
    
    def merge_similar_facts(self, similarity_threshold: float = 0.8) -> int:
        """Merge similar facts to reduce redundancy"""
        with self.lock:
            if self.backend == "networkx":
                merged_count = 0
                facts = [node for node in self.nodes.values() if node.type == MemoryType.FACT]
                facts_to_remove = set()
                
                for i, fact1 in enumerate(facts):
                    if fact1.id in facts_to_remove:  # Already marked for removal
                        continue
                    
                    for fact2 in facts[i+1:]:
                        if fact2.id in facts_to_remove:  # Already marked for removal
                            continue
                        
                        # Check similarity
                        similarity = self._calculate_fact_similarity(fact1, fact2)
                        if similarity >= similarity_threshold:
                            # Merge facts (keep the one with higher confidence)
                            if fact1.confidence >= fact2.confidence:
                                # Update fact1 with combined content
                                combined_content = f"{fact1.content} (Also: {fact2.content})"
                                self.update_fact(fact1.id, combined_content, 
                                               new_confidence=max(fact1.confidence, fact2.confidence))
                                # Mark fact2 for removal
                                facts_to_remove.add(fact2.id)
                            else:
                                # Update fact2 with combined content
                                combined_content = f"{fact2.content} (Also: {fact1.content})"
                                self.update_fact(fact2.id, combined_content, 
                                               new_confidence=max(fact1.confidence, fact2.confidence))
                                # Mark fact1 for removal
                                facts_to_remove.add(fact1.id)
                                break
                            
                            merged_count += 1
                
                # Remove marked facts
                for fact_id in facts_to_remove:
                    self.delete_fact(fact_id)
                
                return merged_count
            else:
                return self._merge_similar_facts_in_neo4j(similarity_threshold)
    
    def _calculate_fact_similarity(self, fact1: MemoryNode, fact2: MemoryNode) -> float:
        """Calculate similarity between two facts"""
        if self.embedding_model:
            # Use semantic similarity
            try:
                emb1 = self._get_node_embedding(fact1)
                emb2 = self._get_node_embedding(fact2)
                if emb1 is not None and emb2 is not None:
                    return self._cosine_similarity(emb1, emb2)
            except Exception:
                pass
        
        # Fallback to basic text similarity
        words1 = set(fact1.content.lower().split())
        words2 = set(fact2.content.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        # Also check for substring similarity
        content1 = fact1.content.lower()
        content2 = fact2.content.lower()
        
        # If one is substring of another, give high similarity
        if content1 in content2 or content2 in content1:
            return 0.9
        
        # Check for key word overlap
        key_words1 = {w for w in words1 if len(w) > 3}  # Focus on longer words
        key_words2 = {w for w in words2 if len(w) > 3}
        
        if key_words1 and key_words2:
            key_intersection = len(key_words1.intersection(key_words2))
            key_union = len(key_words1.union(key_words2))
            if key_union > 0:
                key_similarity = key_intersection / key_union
                # Combine with general similarity
                general_similarity = intersection / union if union > 0 else 0.0
                return max(general_similarity, key_similarity)
        
        return intersection / union if union > 0 else 0.0
    
    def export_memory_graph(self, format: str = "json") -> str:
        """Export memory graph in various formats"""
        with self.lock:
            if format.lower() == "json":
                data = {
                    'nodes': [node.to_dict() for node in self.nodes.values()],
                    'relationships': [rel.to_dict() for rel in self.relationships.values()],
                    'stats': self.stats,
                    'export_timestamp': datetime.now().isoformat()
                }
                return json.dumps(data, indent=2, ensure_ascii=False)
            elif format.lower() == "graphml" and NETWORKX_AVAILABLE:
                # Export as GraphML for visualization tools
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.graphml', delete=False)
                nx.write_graphml(self.graph, temp_file.name)
                with open(temp_file.name, 'r') as f:
                    content = f.read()
                os.unlink(temp_file.name)
                return content
            else:
                raise ValueError(f"Unsupported export format: {format}")
    
    def import_memory_graph(self, data: str, format: str = "json", clear_existing: bool = False) -> int:
        """Import memory graph from various formats"""
        with self.lock:
            if clear_existing:
                self.clear_memory()
            
            if format.lower() == "json":
                try:
                    import_data = json.loads(data)
                    
                    # Import nodes
                    node_count = 0
                    for node_data in import_data.get('nodes', []):
                        node = MemoryNode.from_dict(node_data)
                        self.nodes[node.id] = node
                        self.graph.add_node(node.id, 
                                          id=node.id,
                                          type=node.type.value,
                                          content=node.content,
                                          metadata=node.metadata,
                                          created_at=node.created_at.isoformat(),
                                          updated_at=node.updated_at.isoformat(),
                                          confidence=node.confidence,
                                          access_count=node.access_count)
                        node_count += 1
                    
                    # Import relationships
                    rel_count = 0
                    for rel_data in import_data.get('relationships', []):
                        rel = MemoryRelationship.from_dict(rel_data)
                        rel_key = f"{rel.source_id}_{rel.target_id}_{rel.relationship_type.value}"
                        self.relationships[rel_key] = rel
                        self.graph.add_edge(rel.source_id, rel.target_id, 
                                          relationship_type=rel.relationship_type.value,
                                          source_id=rel.source_id,
                                          target_id=rel.target_id,
                                          strength=rel.strength,
                                          created_at=rel.created_at.isoformat(),
                                          metadata=rel.metadata)
                        rel_count += 1
                    
                    self._save_to_disk()
                    return node_count
                except Exception as e:
                    logger.error(f"Error importing memory graph: {e}")
                    return 0
            else:
                raise ValueError(f"Unsupported import format: {format}")
    
    # Neo4j helper methods for new functions
    def _get_recent_facts_from_neo4j(self, days: int, limit: int) -> List[MemoryNode]:
        """Get recent facts from Neo4j"""
        with self.driver.session() as session:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            result = session.run("""
                MATCH (n:MemoryNode)
                WHERE n.type = 'fact' AND n.created_at >= $cutoff_date
                RETURN n ORDER BY n.created_at DESC LIMIT $limit
            """, cutoff_date=cutoff_date, limit=limit)
            return [MemoryNode.from_dict(record["n"]) for record in result]
    
    def _get_facts_by_confidence_from_neo4j(self, min_confidence: float, limit: int) -> List[MemoryNode]:
        """Get facts by confidence from Neo4j"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:MemoryNode)
                WHERE n.type = 'fact' AND n.confidence >= $min_confidence
                RETURN n ORDER BY n.confidence DESC LIMIT $limit
            """, min_confidence=min_confidence, limit=limit)
            return [MemoryNode.from_dict(record["n"]) for record in result]
    
    def _merge_similar_facts_in_neo4j(self, similarity_threshold: float) -> int:
        """Merge similar facts in Neo4j (simplified implementation)"""
        # This would require more complex Cypher queries
        # For now, return 0 as placeholder
        return 0

    def _update_stats(self):
        """Update memory statistics"""
        if self.backend == "networkx":
            self.stats['total_nodes'] = len(self.nodes)
            self.stats['total_relationships'] = len(self.relationships)
            
            # Count by type
            type_counts = {}
            for node in self.nodes.values():
                node_type = node.type.value
                type_counts[node_type] = type_counts.get(node_type, 0) + 1
            self.stats['memory_types'] = type_counts


# Test driver code
def test_graph_memory():
    """Test the graph memory system"""
    print("=== Testing Graph Memory System ===\n")
    
    # Initialize memory manager
    memory = GraphMemoryManager(backend="networkx")
    
    # Test 1: Add facts
    print("1. Adding facts...")
    fact1_id = memory.add_fact("Python is a programming language", "technical", {"source": "general_knowledge"})
    fact2_id = memory.add_fact("Machine learning uses algorithms to learn patterns", "technical", {"source": "ai_knowledge"})
    fact3_id = memory.add_fact("The user prefers technical explanations", "personal", {"source": "user_preference"})
    fact4_id = memory.add_fact("Python is used for data science", "technical", {"source": "data_science"})
    fact5_id = memory.add_fact("The user works in software development", "personal", {"source": "user_profile"})
    
    print(f"   Added facts: {fact1_id}, {fact2_id}, {fact3_id}, {fact4_id}, {fact5_id}")
    
    # Test 2: Get all facts
    print("\n2. Getting all facts...")
    all_facts = memory.get_all_facts()
    for fact in all_facts:
        print(f"   - {fact.content} (Type: {fact.metadata.get('fact_type')})")
    
    # Test 3: Add relationships
    print("\n3. Adding relationships...")
    memory.add_relationship(fact1_id, fact2_id, RelationshipType.RELATED_TO, {"reason": "both are technical"})
    memory.add_relationship(fact2_id, fact3_id, RelationshipType.RELATED_TO, {"reason": "user preference affects ML usage"})
    memory.add_relationship(fact1_id, fact4_id, RelationshipType.RELATED_TO, {"reason": "Python is used in both"})
    memory.add_relationship(fact3_id, fact5_id, RelationshipType.RELATED_TO, {"reason": "both are personal preferences"})
    
    # Test 4: Get related facts
    print("\n4. Getting related facts...")
    related = memory.get_related_facts(fact1_id)
    for fact, rel_type, strength in related:
        print(f"   - {fact.content} (Relationship: {rel_type.value}, Strength: {strength})")
    
    # Test 5: Search facts
    print("\n5. Searching facts...")
    search_results = memory.search_facts("programming language", limit=5)
    for fact, similarity in search_results:
        print(f"   - {fact.content} (Similarity: {similarity:.3f})")
    
    # Test 6: Update fact
    print("\n6. Updating fact...")
    memory.update_fact(fact1_id, "Python is a high-level programming language", new_confidence=0.9)
    updated_fact = memory.get_fact(fact1_id)
    print(f"   Updated: {updated_fact.content} (Confidence: {updated_fact.confidence})")
    
    # Test 7: Get facts by type
    print("\n7. Getting facts by type...")
    technical_facts = memory.get_facts_by_type("technical")
    personal_facts = memory.get_facts_by_type("personal")
    print(f"   Technical facts: {len(technical_facts)}")
    print(f"   Personal facts: {len(personal_facts)}")
    
    # Test 8: Get facts by confidence
    print("\n8. Getting facts by confidence...")
    high_conf_facts = memory.get_facts_by_confidence(min_confidence=0.8)
    print(f"   High confidence facts: {len(high_conf_facts)}")
    for fact in high_conf_facts:
        print(f"   - {fact.content} (Confidence: {fact.confidence})")
    
    # Test 9: Get recent facts
    print("\n9. Getting recent facts...")
    recent_facts = memory.get_recent_facts(days=1)
    print(f"   Recent facts (last 1 day): {len(recent_facts)}")
    
    # Test 10: Test duplicate prevention
    print("\n10. Testing duplicate prevention...")
    duplicate_id = memory.add_fact("Python is a high-level programming language", "technical")
    print(f"   Duplicate fact ID: {duplicate_id} (should be same as {fact1_id})")
    
    # Test 11: Add similar facts for merging test
    print("\n11. Adding similar facts for merging...")
    similar_fact1_id = memory.add_fact("Python is a programming language for beginners", "technical")
    similar_fact2_id = memory.add_fact("Python programming language is easy to learn", "technical")
    
    # Test 12: Merge similar facts
    print("\n12. Merging similar facts...")
    merged_count = memory.merge_similar_facts(similarity_threshold=0.5)  # Lower threshold for testing
    print(f"   Merged {merged_count} facts")
    
    # Show remaining facts after merging
    remaining_facts = memory.get_all_facts()
    print(f"   Remaining facts after merging: {len(remaining_facts)}")
    for fact in remaining_facts:
        print(f"     - {fact.content}")
    
    # Test 13: Get statistics
    print("\n13. Memory statistics...")
    stats = memory.get_memory_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test 14: Export memory
    print("\n14. Exporting memory...")
    export_data = memory.export_memory_graph("json")
    print(f"   Export data length: {len(export_data)} characters")
    
    # Test 15: Import memory
    print("\n15. Testing import...")
    # Create a new memory instance
    memory2 = GraphMemoryManager(backend="networkx", db_path="test_import.db")
    imported_count = memory2.import_memory_graph(export_data, clear_existing=True)
    print(f"   Imported {imported_count} nodes")
    
    # Test 16: Delete fact
    print("\n16. Deleting fact...")
    memory.delete_fact(fact3_id)
    deleted_fact = memory.get_fact(fact3_id)
    print(f"   Deleted fact exists: {deleted_fact is not None}")
    
    # Test 17: Backup
    print("\n17. Creating backup...")
    backup_success = memory.backup_memory("memory_backup.json")
    print(f"   Backup successful: {backup_success}")
    
    # Test 18: Clear specific memory type
    print("\n18. Clearing specific memory type...")
    cleared_count = memory.clear_memory(MemoryType.FACT)
    print(f"   Cleared {cleared_count} facts")
    
    print("\n=== Test completed ===")


def test_chat_integration():
    """Test integration with chat-like scenarios"""
    print("\n=== Testing Chat Integration ===\n")
    
    memory = GraphMemoryManager(backend="networkx", db_path="chat_memory.db")
    
    # Simulate a conversation
    print("Simulating a conversation...")
    
    # User asks about Python
    user_query = "What is Python?"
    print(f"User: {user_query}")
    
    # Search for relevant facts
    relevant_facts = memory.search_facts(user_query, limit=3)
    if relevant_facts:
        print("Relevant facts found:")
        for fact, similarity in relevant_facts:
            print(f"  - {fact.content} (Relevance: {similarity:.3f})")
    else:
        print("No relevant facts found, adding new fact...")
        memory.add_fact("Python is a high-level programming language", "technical", {"context": "user_query"})
    
    # User asks follow-up
    user_query2 = "What can I do with Python?"
    print(f"\nUser: {user_query2}")
    
    # Search again
    relevant_facts2 = memory.search_facts(user_query2, limit=3)
    if relevant_facts2:
        print("Relevant facts found:")
        for fact, similarity in relevant_facts2:
            print(f"  - {fact.content} (Relevance: {similarity:.3f})")
    
    # Add more context
    memory.add_fact("Python can be used for web development, data analysis, and AI", "technical", {"context": "capabilities"})
    memory.add_fact("The user is interested in Python programming", "personal", {"context": "user_interest"})
    
    print("\nChat integration test completed!")


if __name__ == "__main__":
    test_graph_memory()
    test_chat_integration()
