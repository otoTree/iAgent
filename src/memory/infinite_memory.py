# src/memory/infinite_memory.py
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime, timedelta
import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity # Commented out
import redis
import chromadb
# from neo4j import GraphDatabase # Commented out
import json
import uuid # Added for memory IDs

class InfiniteMemorySystem:
    """无限记忆系统实现"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Short-term memory - Redis
        try:
            self.short_term = redis.Redis(
                host=config.get('redis_host', 'localhost'),
                port=config.get('redis_port', 6379),
                decode_responses=True
            )
            self.short_term.ping() # Check connection
            print("Successfully connected to Redis for short-term memory.")
        except redis.exceptions.ConnectionError as e:
            print(f"Error connecting to Redis for short-term memory: {e}. Short-term memory will be disabled.")
            self.short_term = None

        # Long-term memory - Vector database (ChromaDB)
        try:
            self.vector_db_client = chromadb.Client() # Or use PersistentClient for on-disk storage
            # self.vector_db_client = chromadb.PersistentClient(path="./chroma_db_data") # Example for persistent
            self.knowledge_collection = self.vector_db_client.get_or_create_collection(
                name="long_term_knowledge",
                metadata={"hnsw:space": "cosine"} # Cosine similarity is common for embeddings
            )
            print("Successfully connected/initialized ChromaDB for long-term memory.")
        except Exception as e: # Catching a broad exception as chromadb setup can have various issues
            print(f"Error initializing ChromaDB for long-term memory: {e}. Long-term memory will be disabled.")
            self.vector_db_client = None
            self.knowledge_collection = None
        
        # Episodic memory - Simplified list for now
        self.episodic_memory: List[Dict[str, Any]] = [] 
        print("Episodic memory initialized (in-memory list).")
        
        # Skill memory - In-memory dictionary
        self.skill_memory: Dict[str, Dict[str, Any]] = {}
        print("Skill memory initialized (in-memory dictionary).")
        
        # Knowledge Graph - Neo4j (connection placeholder)
        self.graph_driver = None
        if config.get('neo4j_uri') and config.get('neo4j_user') and config.get('neo4j_password'):
            try:
                # self.graph_driver = GraphDatabase.driver( # Commented out
                #     config['neo4j_uri'],
                #     auth=(config['neo4j_user'], config['neo4j_password'])
                # )
                # self.graph_driver.verify_connectivity() # Check connection
                print(f"Placeholder: Neo4j connection would be established to {config['neo4j_uri']}.")
                # For now, we are not actually connecting to Neo4j to avoid dependency error.
                # If you have Neo4j running and python driver installed, uncomment above lines.
            except Exception as e: # Catching a broad exception for driver issues
                print(f"Error connecting to Neo4j: {e}. Knowledge graph features will be limited.")
                self.graph_driver = None
        else:
            print("Neo4j config not fully provided. Knowledge graph features will be limited.")

    async def _generate_embedding(self, text: str) -> List[float]:
        """Placeholder: Generates embedding for text. Uses a very simple hashing approach for now."""
        print(f"Placeholder: Generating embedding for text: '{text[:50]}...'")
        # In a real system, use a sentence transformer or other embedding model.
        # e.g., from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer('all-MiniLM-L6-v2')
        # return model.encode(text).tolist()
        
        # Simple placeholder: average of character ordinals, fixed size (e.g., 10)
        if not text: return [0.0] * 10
        hash_val = sum(ord(c) for c in text)
        embedding = [float(hash_val % (i+100)) for i in range(10)]
        # Normalize (very crudely)
        norm = np.linalg.norm(embedding)
        return (embedding / norm).tolist() if norm > 0 else [0.0] * 10


    async def store(self, memory_type: str, data: Dict[str, Any], memory_id: Optional[str] = None) -> str:
        """存储记忆"""
        if memory_id is None:
             memory_id = f"{memory_type}_{uuid.uuid4().hex}" # Ensure unique ID using uuid

        timestamp = datetime.now()
        data_to_store = {**data, 'timestamp': timestamp.isoformat(), 'id': memory_id}

        print(f"Storing memory: Type='{memory_type}', ID='{memory_id}'")

        if memory_type == "short_term":
            if self.short_term:
                self.short_term.setex(
                    memory_id,
                    timedelta(hours=data.get("expiry_hours", 24)), # Allow custom expiry
                    json.dumps(data_to_store)
                )
            else:
                print("Warning: Short-term memory (Redis) not available. Cannot store.")
                return "" # Indicate failure
            
        elif memory_type == "long_term":
            if self.knowledge_collection:
                content = data.get('content', '')
                if not content:
                    print("Warning: Long-term memory content is empty. Not storing.")
                    return ""
                embedding = await self._generate_embedding(content)
                metadata = data.get('metadata', {})
                metadata['original_id'] = memory_id # Store original ID if needed
                metadata['timestamp'] = timestamp.isoformat()

                self.knowledge_collection.add(
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[metadata], # Ensure metadata is a list of dicts
                    ids=[memory_id] # Chroma IDs must be unique strings
                )
            else:
                print("Warning: Long-term memory (ChromaDB) not available. Cannot store.")
                return ""
            
        elif memory_type == "episodic":
            # Episodic memory includes context, actions, outcome
            self.episodic_memory.append({
                'id': memory_id,
                'timestamp': timestamp,
                'context': data.get('context'),
                'actions': data.get('actions'),
                'outcome': data.get('outcome')
            })
            # Optional: Limit size of in-memory episodic memory
            if len(self.episodic_memory) > 10000: # Example limit
                self.episodic_memory.pop(0)

        elif memory_type == "skill":
            skill_data = {
                'id': memory_id,
                'name': data.get('name'),
                'code': data.get('code'),
                'description': data.get('description'),
                'usage_count': data.get('usage_count', 0),
                'success_rate': data.get('success_rate', 0.0),
                'timestamp': timestamp.isoformat()
            }
            self.skill_memory[memory_id] = skill_data
            
        elif memory_type == "relationship":
            if self.graph_driver:
                # await self._store_relationship(data_to_store) # Actual call commented
                print(f"Placeholder: Storing relationship to Neo4j: {data_to_store}")
            else:
                print("Warning: Knowledge Graph (Neo4j) not available. Cannot store relationship.")
                return ""
        else:
            print(f"Warning: Unknown memory type '{memory_type}'. Not storing.")
            return ""
            
        return memory_id
        
    async def retrieve(self, query: str, memory_types: Optional[List[str]] = None, 
                      top_k: int = 5) -> List[Dict[str, Any]]:
        """检索相关记忆"""
        if memory_types is None:
            memory_types = ["short_term", "long_term", "episodic", "skill", "relationship"]
            
        all_results: List[Dict[str, Any]] = []
        
        print(f"Retrieving memory: Query='{query[:50]}...', Types={memory_types}, TopK={top_k}")

        # 1. Short-term memory retrieval (exact match on keys or simple scan)
        if "short_term" in memory_types and self.short_term:
            # This is a simplistic retrieval. Real retrieval might involve query against content.
            try:
                # Example: scan for keys matching a pattern derived from the query (if applicable)
                # For now, let's assume we search for a key that might be the query itself if it's an ID
                # Or, one might iterate a few recent keys and check content.
                # This part highly depends on how short_term keys are structured.
                # For a generic query, short-term might be less useful unless query is an ID.
                retrieved_item = self.short_term.get(query) # If query is an ID
                if retrieved_item:
                    item_data = json.loads(retrieved_item)
                    all_results.append({**item_data, "_score": 1.0, "_source": "short_term_direct"})
                
                # A more general approach (but potentially slow):
                # Iterate some keys (e.g., recent ones if pattern known)
                # For this placeholder, we'll skip more complex short-term search.
                print("Placeholder: Short-term memory search is basic (direct key lookup or needs specific logic).")

            except Exception as e:
                print(f"Error retrieving from short-term memory: {e}")

        # 2. Long-term memory vector retrieval
        if "long_term" in memory_types and self.knowledge_collection:
            try:
                query_embedding = await self._generate_embedding(query)
                vector_results = self.knowledge_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"] # Request distances for scoring
                )
                # print(f"Chroma query results: {vector_results}")
                formatted_results = self._format_vector_results(vector_results, query)
                all_results.extend(formatted_results)
            except Exception as e:
                print(f"Error retrieving from long-term memory (ChromaDB): {e}")

        # 3. Episodic memory retrieval (keyword search or similarity on context/outcome)
        if "episodic" in memory_types:
            try:
                episodic_results = await self._search_episodic_placeholder(query, top_k)
                all_results.extend(episodic_results)
            except Exception as e:
                print(f"Error retrieving from episodic memory: {e}")

        # 4. Skill memory retrieval (keyword search on description or name)
        if "skill" in memory_types:
            try:
                skill_results = await self._search_skills_placeholder(query, top_k)
                all_results.extend(skill_results)
            except Exception as e:
                print(f"Error retrieving from skill memory: {e}")

        # 5. Relationship / Knowledge Graph retrieval
        if "relationship" in memory_types and self.graph_driver:
            try:
                # kg_results = await self._search_knowledge_graph(query, top_k) # Actual call
                # all_results.extend(kg_results)
                print(f"Placeholder: Knowledge graph search for '{query}' would be performed here.")
            except Exception as e:
                print(f"Error retrieving from knowledge graph: {e}")

        # 6. Sort, deduplicate, and take top_k
        # Sort by score (descending, so higher is better)
        # This simple scoring needs to be refined. Chroma's distance is lower-is-better.
        # Other sources might need different scoring.
        # For now, we assume _score is higher-is-better.
        all_results.sort(key=lambda x: x.get("_score", 0), reverse=True)
        
        # Deduplicate based on 'id' or a content hash if available
        seen_ids = set()
        final_results = []
        for res in all_results:
            res_id = res.get('id') or res.get('metadata', {}).get('original_id') # Check both places
            if res_id and res_id not in seen_ids:
                final_results.append(res)
                seen_ids.add(res_id)
            elif not res_id and res.get('content'): # Fallback for items without clear ID, less reliable
                 # This could be slow or inaccurate for deduplication
                 content_to_check = res.get('content', str(res)) 
                 if content_to_check not in seen_ids:
                      final_results.append(res)
                      seen_ids.add(content_to_check) 
            elif not res_id: # If no ID and no content, just add if not too many already
                if len(final_results) < top_k * 2: # Avoid too many unidentifiable results
                    final_results.append(res)


        return final_results[:top_k]

    def _format_vector_results(self, vector_results: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        """Formats results from ChromaDB query, converting distance to a score."""
        formatted = []
        ids = vector_results.get('ids', [[]])[0]
        documents = vector_results.get('documents', [[]])[0]
        metadatas = vector_results.get('metadatas', [[]])[0]
        distances = vector_results.get('distances', [[]])[0]

        for i, doc_id in enumerate(ids):
            distance = distances[i] if distances and i < len(distances) else None
            # Convert cosine distance to similarity score (0 to 1, higher is better)
            # Cosine distance from Chroma is 1 - similarity for L2/IP, or actual distance for cosine.
            # If metadata hnsw:space is "cosine", distance is 1-cos_sim for normalized vectors.
            # So, score = 1 - distance (if distance is 0 to 2 for cosine)
            # Or if distance is already cos_sim (some setups), then score = distance
            # Assuming distance is true cosine distance (0-2 range), so 1-dist/2 or similar.
            # For "cosine" space in Chroma, distance is sqrt(2-2*cos_similarity).
            # A simpler approach: score = 1 / (1 + distance) if distance is always positive.
            score = 1.0 / (1.0 + distance) if distance is not None else 0.0

            formatted.append({
                "id": doc_id,
                "content": documents[i] if documents and i < len(documents) else None,
                "metadata": metadatas[i] if metadatas and i < len(metadatas) else None,
                "score": score,
                "_source": "long_term_vector"
            })
        return formatted
        
    async def _search_episodic_placeholder(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Placeholder for searching episodic memory. Simple keyword matching for now."""
        results = []
        query_lower = query.lower()
        for episode in self.episodic_memory:
            context = episode.get('context', '')
            actions = str(episode.get('actions', '')) # Convert actions list/dict to string for simple search
            outcome = episode.get('outcome', '')
            
            score = 0
            if query_lower in (context.lower() if context else ''): score += 0.5
            if query_lower in (actions.lower() if actions else ''): score += 0.3
            if query_lower in (outcome.lower() if outcome else ''): score += 0.2
            
            if score > 0:
                results.append({**episode, "_score": score, "_source": "episodic_placeholder"})
        
        results.sort(key=lambda x: x["_score"], reverse=True)
        return results[:top_k]

    async def _search_skills_placeholder(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Placeholder for searching skill memory. Simple keyword matching on name/description."""
        results = []
        query_lower = query.lower()
        for skill_id, skill_data in self.skill_memory.items():
            name = skill_data.get('name', '')
            description = skill_data.get('description', '')
            
            score = 0
            if query_lower in (name.lower() if name else ''): score += 0.6
            if query_lower in (description.lower() if description else ''): score += 0.4
            
            if score > 0:
                results.append({**skill_data, "id": skill_id, "_score": score, "_source": "skill_placeholder"})
        
        results.sort(key=lambda x: x["_score"], reverse=True)
        return results[:top_k]

    async def _store_relationship(self, data: Dict[str, Any]):
        """Placeholder for storing relationships in Neo4j."""
        # Example: data = {"source_node": {"label": "User", "id": "user1"}, 
        #                   "target_node": {"label": "Interest", "name": "AI"}, 
        #                   "relationship_type": "HAS_INTEREST"}
        print(f"Placeholder: Storing relationship in Neo4j: {data}")
        # Actual implementation would use self.graph_driver.execute_query(...)

    async def _search_knowledge_graph(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Placeholder for searching the knowledge graph."""
        print(f"Placeholder: Searching knowledge graph for: {query}")
        # Actual implementation would parse query, build Cypher, and run against Neo4j
        return []

    async def _consolidate_memories(self):
        """记忆整合与压缩 (Placeholder)"""
        print("Placeholder: Starting memory consolidation...")
        if not self.short_term or not self.knowledge_collection:
            print("Memory consolidation skipped: Redis or ChromaDB not available.")
            return

        # Example: Move important short-term memories to long-term
        # This requires a way to list keys from Redis, which can be performance intensive (SCAN)
        # For simplicity, let's assume a hypothetical function get_all_short_term_data()
        # In reality, you might have specific patterns for keys to scan.
        
        # This part is highly conceptual and needs a robust implementation strategy
        # For instance, iterating all keys in Redis for a large DB is not advisable.
        # A better approach might be to log candidate keys for consolidation elsewhere.
        
        # Conceptual:
        # candidate_keys = self.short_term.scan_iter("short_term_*") # Example pattern
        # for key in candidate_keys:
        #     try:
        #         raw_data = self.short_term.get(key)
        #         if raw_data:
        #             data = json.loads(raw_data)
        #             importance = await self._calculate_importance(data) # Importance calculation needed
                    
        #             if importance > 0.7:  # Importance threshold
        #                 content_to_store = data.get('content', data) # Or structure specifically for long term
        #                 if isinstance(content_to_store, dict) and 'content' not in content_to_store:
        #                      # Try to find a meaningful string part if 'content' field is missing
        #                      main_content = next((v for v in content_to_store.values() if isinstance(v, str)), None)
        #                 elif isinstance(content_to_store, str):
        #                      main_content = content_to_store
        #                 else:
        #                      main_content = data.get('content')


        #                 if main_content and isinstance(main_content, str):
        #                     await self.store("long_term", {
        #                         "content": main_content,
        #                         "metadata": {"source": "consolidated_short_term", **data.get("metadata", {})},
        #                         # Pass original ID for deduplication or linking if necessary
        #                         "original_id": data.get("id", key) 
        #                     })
        #                     self.short_term.delete(key)
        #                     print(f"Consolidated short-term memory {key} to long-term.")
        #                 else:
        #                     print(f"Skipping consolidation for {key}, no suitable string content found.")
        #     except Exception as e:
        #         print(f"Error during consolidation of key {key}: {e}")

        # Placeholder for compressing similar memories (e.g., in long-term or episodic)
        # await self._compress_similar_memories_placeholder()
        print("Placeholder: Memory consolidation finished.")
        
    async def _calculate_importance(self, memory_data: Dict[str, Any]) -> float:
        """计算记忆的重要性 (Placeholder)"""
        # Simple placeholder: access_count and recency (if timestamp is available)
        # This needs a more sophisticated model in a real system.
        timestamp_str = memory_data.get('timestamp')
        recency_score = 0.0
        if timestamp_str:
            try:
                mem_ts = datetime.fromisoformat(timestamp_str)
                recency_score = 1.0 / (1.0 + (datetime.now() - mem_ts).days)
            except ValueError: # Invalid timestamp format
                pass
        
        # Example factors (these would often come from interaction metadata)
        access_count = memory_data.get('metadata', {}).get('access_count', 0)
        emotional_score = memory_data.get('metadata', {}).get('emotional_score', 0.5) # e.g. from sentiment
        utility_score = memory_data.get('metadata', {}).get('utility_score', 0.5) # e.g. if memory led to success

        # Weighted average (example weights)
        importance = (
            (access_count / 10.0 * 0.3) +  # Normalize access count, cap effect
            (recency_score * 0.2) +
            (emotional_score * 0.2) +
            (utility_score * 0.3)
        )
        return min(1.0, max(0.0, importance)) # Ensure score is between 0 and 1


    async def _compress_similar_memories_placeholder(self):
        """Placeholder for compressing or summarizing similar memories."""
        print("Placeholder: Compressing similar memories (not implemented).")
        # This could involve:
        # 1. Querying for highly similar items in vector DB.
        # 2. Using an LLM to summarize or merge them.
        # 3. Updating/replacing the old items with the new compressed one.

    async def _clean_expired_memories(self): # Added this method, called by main loop
        """ Placeholder for cleaning memories that are not handled by Redis TTL e.g. episodic/skill """
        print("Placeholder: Cleaning expired memories (beyond Redis TTL).")
        # Example for episodic (if they had an expiry logic)
        # current_time = datetime.now()
        # self.episodic_memory = [
        #    e for e in self.episodic_memory 
        #    if not hasattr(e, 'expires_at') or e.expires_at > current_time
        # ]
        pass

    async def _optimize_knowledge_graph(self): # Added this method, called by main loop
        """ Placeholder for optimizing knowledge graph (e.g. pruning, inferring new links) """
        print("Placeholder: Optimizing knowledge graph.")
        pass

```
