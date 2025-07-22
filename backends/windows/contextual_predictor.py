import os
import re
import json
import faiss
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from collections import defaultdict

class AdaptiveContextualPredictor:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.dimension = 384  
        self.faiss_indices = {}
        self.metadata_store = defaultdict(list)
        self.interaction_history = defaultdict(list)

    def preprocess_text(self, text: str):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def get_suggestions(self, agent_type: str, partial_input: str, max_suggestions=5):
        """
        Enhanced get_suggestions method with:
        1. Unique suggestions only
        2. Proper confidence scoring
        3. Filter for relevant suggestions
        4. Fallback suggestions if needed
        """
        if agent_type not in self.faiss_indices:
            return []
        
        # Embed the partial input
        processed_input = self.preprocess_text(partial_input)
        query_emb = self._get_embeddings([processed_input])
        
        # Get more neighbors than needed to account for duplicates
        index = self.faiss_indices[agent_type]
        k = min(max_suggestions * 3, len(self.metadata_store[agent_type]))  # Get 3x more for filtering
        distances, neighbors = index.search(query_emb, k)
        
        # Process and deduplicate suggestions
        seen_suggestions = set()
        suggestions = []
        all_meta = self.metadata_store[agent_type]
        
        for rank, (idx, distance) in enumerate(zip(neighbors[0], distances[0])):
            item = all_meta[idx]
            suggested_query = item["prompt"]
            
            # Skip if we've seen this suggestion or if it's too similar to what we've seen
            if suggested_query in seen_suggestions:
                continue
                
            # Use sigmoid function for better confidence scoring
            confidence = 1 / (1 + np.exp(distance * 0.1))  # Adjusted scaling factor
            
            # Filter out irrelevant suggestions (confidence too low)
            if confidence < 0.1:  # Adjusted threshold
                continue
            
            suggestions.append({
                "suggested_query": suggested_query,
                "confidence": float(confidence)
            })
            seen_suggestions.add(suggested_query)
            
            if len(suggestions) >= max_suggestions:
                break
        
        # If we don't have enough suggestions, add fallback suggestions
        if len(suggestions) < max_suggestions:
            fallback_suggestions = self._get_fallback_suggestions(agent_type, processed_input)
            for sugg in fallback_suggestions:
                if len(suggestions) >= max_suggestions:
                    break
                if sugg["suggested_query"] not in seen_suggestions:
                    suggestions.append(sugg)
                    seen_suggestions.add(sugg["suggested_query"])
        
        return suggestions

    def _get_fallback_suggestions(self, agent_type: str, partial_input: str):
        """Provide fallback suggestions based on the agent type and input"""
        fallbacks = {
            "legal": [
                "summarize this legal document",
                "analyze this contract for risks",
                "review the legal implications",
                "explain the terms and conditions",
                "check for compliance issues"
            ],
            "meetings": [
                "summarize the meeting discussion",
                "list all action items",
                "highlight key decisions made",
                "provide meeting overview",
                "extract important deadlines"
            ],
            "email": [
                "summarize this email thread",
                "extract action items from emails",
                "provide email context",
                "list key points discussed",
                "compile email decisions"
            ]
        }
        
        # Return fallbacks with lower confidence
        return [
            {
                "suggested_query": sugg,
                "confidence": 0.3  # Lower confidence for fallback suggestions
            }
            for sugg in fallbacks.get(agent_type, [])
        ]

    def update_user_query(self, agent_type: str, user_input: str):
        processed_input = self.preprocess_text(user_input)
        
        # Avoid duplicates in metadata store
        if not any(item["prompt"] == processed_input for item in self.metadata_store[agent_type]):
            self.metadata_store[agent_type].append({
                "prompt": processed_input,
                "timestamp": datetime.now().isoformat()
            })
            
            self.interaction_history[agent_type].append({
                "input": processed_input,
                "timestamp": datetime.now().isoformat()
            })
            
            # Rebuild the FAISS index for this agent
            self._rebuild_faiss_index(agent_type)

    def _rebuild_faiss_index(self, agent_type: str):
        queries = [item["prompt"] for item in self.metadata_store[agent_type]]
        if not queries:
            return
        
        # Create a new FlatL2 index
        index = faiss.IndexFlatL2(self.dimension)
        
        # Embed all queries
        embs = self._get_embeddings(queries)
        
        # Add to the index
        index.add(embs)
        
        self.faiss_indices[agent_type] = index
    
    def _get_embeddings(self, texts):
        embs = self.embedding_model.encode(texts, convert_to_numpy=True)
        return embs.astype(np.float32)
    
    def get_suggestions(self, agent_type: str, partial_input: str, max_suggestions=5):
        """
        Given a user's partial input, find the top-K similar past queries.
        Return them as "suggestions" for auto-completion.
        """
        if agent_type not in self.faiss_indices:
            return []
        
        # Embed the partial input
        processed_input = self.preprocess_text(partial_input)
        query_emb = self._get_embeddings([processed_input])  # shape (1, dimension)
        
        index = self.faiss_indices[agent_type]
        
        # Search the index for nearest neighbors
        distances, neighbors = index.search(query_emb, max_suggestions)
        
        # Prepare suggestions
        suggestions = []
        all_meta = self.metadata_store[agent_type]
        
        for rank, idx in enumerate(neighbors[0]):
            # each idx corresponds to metadata_store[agent_type][idx]
            item = all_meta[idx]
            dist = distances[0][rank]
            
            # Use a simple "confidence" measure (inverse distance)
            confidence = 1.0 / (1.0 + dist)
            
            suggestions.append({
                "suggested_query": item["prompt"],
                "confidence": float(confidence)
            })
        
        return suggestions
    
    def save_state(self, filepath: str):
        """
        Save metadata_store and interaction_history (if needed).
        We won't store the raw FAISS index; we'll rebuild it after loading.
        """
        state = {
            "metadata_store": dict(self.metadata_store),
            "interaction_history": dict(self.interaction_history)
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f)
    
    def load_state(self, filepath: str):
        """
        Load JSON state, then rebuild FAISS indexes for each agent.
        """
        if not os.path.exists(filepath):
            return
        
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
        
        # Rebuild metadata_store
        loaded_meta = state.get("metadata_store", {})
        self.metadata_store = defaultdict(list)
        for agent_type, queries in loaded_meta.items():
            self.metadata_store[agent_type].extend(queries)
        
        # Rebuild interaction_history
        loaded_history = state.get("interaction_history", {})
        self.interaction_history = defaultdict(list)
        for agent_type, items in loaded_history.items():
            self.interaction_history[agent_type].extend(items)
        
        # Now rebuild all FAISS indexes
        self.faiss_indices = {}
        for agent_type in self.metadata_store.keys():
            self._rebuild_faiss_index(agent_type)
