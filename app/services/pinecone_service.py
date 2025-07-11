"""
Pinecone Vector Database Service
Handles vector storage, retrieval, and semantic search operations
"""
from typing import Dict, Any, List, Optional, Union
import logging
import json
import hashlib
import os
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec  # Add this import at the top

logger = logging.getLogger(__name__)

class PineconeService:
    """Handles Pinecone vector database operations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.api_key = self.config.get('api_key') or os.getenv('PINECONE_API_KEY')
        self.environment = self.config.get('environment') or os.getenv('PINECONE_ENVIRONMENT', 'us-east1-gcp')
        self.index_name = self.config.get('index_name', 'dynamic-ai-agent')
        self.dimension = self.config.get('dimension', 1024)  # Default for OpenAI embeddings
        
        # Initialize Pinecone client
        self.pinecone_client = None
        self.index = None
        self._initialize_pinecone()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client and index using the new Pinecone API"""
        try:
            if Pinecone is None or ServerlessSpec is None:
                logger.warning("Pinecone library not available")
                self.pinecone_available = False
                return

            if not self.api_key:
                logger.warning("Pinecone API key not provided")
                self.pinecone_available = False
                return

            # Create Pinecone client instance
            self.pinecone_client = Pinecone(api_key=self.api_key)
            self.pinecone_available = True
            logger.info("Pinecone client initialized successfully (new API)")
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            self.pinecone_available = False

    def create_index(self, index_name: str = None, dimension: int = None, cloud: str = 'aws', region: str = 'us-west-2') -> Dict[str, Any]:
        """
        Create a new Pinecone index using the new API
        """
        if not self.pinecone_available or self.pinecone_client is None:
            return {"error": "Pinecone not available"}

        index_name = index_name or self.index_name or "default-index"
        dimension = dimension or self.dimension or 1024

        try:
            # Check if index already exists
            if hasattr(self.pinecone_client, 'list_indexes') and index_name in self.pinecone_client.list_indexes().names():
                return {
                    "success": True,
                    "message": f"Index '{index_name}' already exists",
                    "index_name": index_name
                }

            if ServerlessSpec is None:
                return {"error": "ServerlessSpec not available in pinecone package"}

            self.pinecone_client.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region)
            )

            logger.info(f"Created Pinecone index: {index_name}")

            return {
                "success": True,
                "message": f"Index '{index_name}' created successfully",
                "index_name": index_name,
                "dimension": dimension
            }

        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            return {"error": f"Index creation failed: {str(e)}"}

    def connect_to_index(self, index_name: str = None):
        """
        Connect to an existing Pinecone index using the new API
        """
        if not self.pinecone_available or self.pinecone_client is None:
            return {"error": "Pinecone not available"}

        index_name = index_name or self.index_name or "default-index"

        try:
            if hasattr(self.pinecone_client, 'list_indexes') and index_name not in self.pinecone_client.list_indexes().names():
                return {"error": f"Index '{index_name}' does not exist"}

            if hasattr(self.pinecone_client, 'Index'):
                self.index = self.pinecone_client.Index(index_name)
                stats = self.index.describe_index_stats()
                logger.info(f"Connected to Pinecone index: {index_name}")
                return {
                    "success": True,
                    "message": f"Connected to index '{index_name}'",
                    "index_name": index_name,
                    "stats": stats
                }
            else:
                return {"error": "Pinecone client does not have Index method"}
        except Exception as e:
            logger.error(f"Error connecting to index: {str(e)}")
            return {"error": f"Index connection failed: {str(e)}"}
    
    def upsert_vectors(self, vectors: List[Dict[str, Any]], namespace: str = None) -> Dict[str, Any]:
        """
        Insert or update vectors in the index
        
        Args:
            vectors: List of vector dictionaries with id, values, and metadata
            namespace: Optional namespace for the vectors
            
        Returns:
            Upsert result
        """
        if not self.index:
            return {"error": "Not connected to any index"}
        
        try:
            # Format vectors for Pinecone
            formatted_vectors = []
            for vector in vectors:
                formatted_vector = {
                    "id": vector["id"],
                    "values": vector["values"]
                }
                
                if "metadata" in vector:
                    formatted_vector["metadata"] = vector["metadata"]
                
                formatted_vectors.append(formatted_vector)
            
            # Upsert vectors
            if namespace:
                response = self.index.upsert(vectors=formatted_vectors, namespace=namespace)
            else:
                response = self.index.upsert(vectors=formatted_vectors)
            
            logger.info(f"Upserted {len(vectors)} vectors")
            
            return {
                "success": True,
                "upserted_count": response.upserted_count,
                "namespace": namespace
            }
            
        except Exception as e:
            logger.error(f"Error upserting vectors: {str(e)}")
            return {"error": f"Vector upsert failed: {str(e)}"}
    
    def upsert_vector(self, vector_id: str, embedding: list, metadata: dict = None, namespace: str = None) -> dict:
        """
        Upsert a single vector (embedding) into Pinecone index.
        Args:
            vector_id: Unique ID for the vector
            embedding: Embedding vector (list of floats)
            metadata: Optional metadata dict
            namespace: Optional namespace
        Returns:
            Upsert result dict
        """
        if not self.index:
            return {"error": "Not connected to any index"}
        try:
            vector = {
                "id": vector_id,
                "values": embedding,
            }
            if metadata:
                vector["metadata"] = metadata
            if namespace:
                response = self.index.upsert(vectors=[vector], namespace=namespace)
            else:
                response = self.index.upsert(vectors=[vector])
            logger.info(f"Upserted vector {vector_id}")
            return {"success": True, "upserted_count": response.upserted_count, "vector_id": vector_id}
        except Exception as e:
            logger.error(f"Error upserting vector {vector_id}: {str(e)}")
            return {"error": f"Vector upsert failed: {str(e)}"}
    
    def query_vectors(self, query_vector: List[float], top_k: int = 10, 
                     namespace: str = None, filter_dict: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Query vectors in the index
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            namespace: Optional namespace to query
            filter_dict: Optional metadata filter
            
        Returns:
            Query results
        """
        if not self.index:
            return {"error": "Not connected to any index"}
        
        try:
            # Build query parameters
            query_params = {
                "vector": query_vector,
                "top_k": top_k,
                "include_metadata": True,
                "include_values": True
            }
            
            if namespace:
                query_params["namespace"] = namespace
            
            if filter_dict:
                query_params["filter"] = filter_dict
            
            # Execute query
            response = self.index.query(**query_params)
            
            # Format results
            results = []
            for match in response.matches:
                result = {
                    "id": match.id,
                    "score": match.score,
                    "values": match.values,
                    "metadata": match.metadata
                }
                results.append(result)
            
            logger.info(f"Query returned {len(results)} results")
            
            return {
                "success": True,
                "results": results,
                "query_params": {
                    "top_k": top_k,
                    "namespace": namespace,
                    "filter": filter_dict
                }
            }
            
        except Exception as e:
            logger.error(f"Error querying vectors: {str(e)}")
            return {"error": f"Vector query failed: {str(e)}"}
    
    def delete_vectors(self, vector_ids: List[str], namespace: str = None) -> Dict[str, Any]:
        """
        Delete vectors from the index
        
        Args:
            vector_ids: List of vector IDs to delete
            namespace: Optional namespace
            
        Returns:
            Deletion result
        """
        if not self.index:
            return {"error": "Not connected to any index"}
        
        try:
            # Delete vectors
            if namespace:
                response = self.index.delete(ids=vector_ids, namespace=namespace)
            else:
                response = self.index.delete(ids=vector_ids)
            
            logger.info(f"Deleted {len(vector_ids)} vectors")
            
            return {
                "success": True,
                "deleted_count": len(vector_ids),
                "namespace": namespace
            }
            
        except Exception as e:
            logger.error(f"Error deleting vectors: {str(e)}")
            return {"error": f"Vector deletion failed: {str(e)}"}
    
    def delete_all_vectors(self, namespace: str = None) -> Dict[str, Any]:
        """Delete all vectors from the index (clear the index)."""
        if not self.index:
            return {"error": "Not connected to any index"}
        try:
            if namespace:
                response = self.index.delete(delete_all=True, namespace=namespace)
            else:
                response = self.index.delete(delete_all=True)
            logger.info("Deleted all vectors from the index" + (f" (namespace: {namespace})" if namespace else ""))
            return {"success": True, "message": "All vectors deleted from the index", "namespace": namespace}
        except Exception as e:
            logger.error(f"Error deleting all vectors: {str(e)}")
            return {"error": f"Failed to delete all vectors: {str(e)}"}
    
    def store_document_chunks(self, document_id: str, chunks: List[Dict[str, Any]], 
                            embeddings: List[List[float]], namespace: str = None) -> Dict[str, Any]:
        """
        Store document chunks with embeddings
        
        Args:
            document_id: Unique document identifier
            chunks: List of text chunks with metadata
            embeddings: List of embedding vectors
            namespace: Optional namespace
            
        Returns:
            Storage result
        """
        if len(chunks) != len(embeddings):
            return {"error": "Number of chunks and embeddings must match"}
        
        try:
            # Prepare vectors for upsert
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{document_id}_chunk_{i}"
                
                metadata = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "text": chunk.get("text", ""),
                    "chunk_size": chunk.get("size", 0),
                    "hash": chunk.get("hash", ""),
                    "created_at": datetime.now().isoformat()
                }
                
                # Add any additional metadata from chunk
                if "metadata" in chunk:
                    metadata.update(chunk["metadata"])
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
            
            # Upsert vectors
            result = self.upsert_vectors(vectors, namespace)
            
            if result.get("success"):
                result["document_id"] = document_id
                result["chunks_stored"] = len(chunks)
            
            return result
            
        except Exception as e:
            logger.error(f"Error storing document chunks: {str(e)}")
            return {"error": f"Document storage failed: {str(e)}"}
    
    def semantic_search(self, query_embedding: List[float], top_k: int = 10,
                       document_id: str = None, namespace: str = None) -> Dict[str, Any]:
        """
        Perform semantic search
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            document_id: Optional document ID filter
            namespace: Optional namespace
            
        Returns:
            Search results
        """
        try:
            # Build filter
            filter_dict = {}
            if document_id:
                filter_dict["document_id"] = document_id
            
            # Query vectors
            result = self.query_vectors(
                query_vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                filter_dict=filter_dict if filter_dict else None
            )
            
            if not result.get("success"):
                return result
            
            # Format search results
            search_results = []
            for match in result["results"]:
                search_result = {
                    "chunk_id": match["id"],
                    "relevance_score": match["score"],
                    "text": match["metadata"].get("text", ""),
                    "document_id": match["metadata"].get("document_id", ""),
                    "chunk_index": match["metadata"].get("chunk_index", 0),
                    "metadata": match["metadata"]
                }
                search_results.append(search_result)
            
            return {
                "success": True,
                "results": search_results,
                "query_info": {
                    "top_k": top_k,
                    "document_filter": document_id,
                    "namespace": namespace
                }
            }
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return {"error": f"Semantic search failed: {str(e)}"}
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        if not self.index:
            return {"error": "Not connected to any index"}
        
        try:
            stats = self.index.describe_index_stats()
            return {
                "success": True,
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {"error": f"Failed to get index stats: {str(e)}"}
    
    def list_indexes(self) -> Dict[str, Any]:
        """List all available indexes"""
        if not self.pinecone_available or self.pinecone_client is None:
            return {"error": "Pinecone not available"}
        
        try:
            indexes = self.pinecone_client.list_indexes()
            return {
                "success": True,
                "indexes": indexes
            }
            
        except Exception as e:
            logger.error(f"Error listing indexes: {str(e)}")
            return {"error": f"Failed to list indexes: {str(e)}"}
    
    def delete_index(self, index_name: str = None) -> Dict[str, Any]:
        """Delete an index"""
        if not self.pinecone_available or self.pinecone_client is None:
            return {"error": "Pinecone not available"}
        
        index_name = index_name or self.index_name or "default-index"
        
        try:
            self.pinecone_client.delete_index(index_name)
            
            # Reset index reference if it was the current one
            if self.index and index_name == self.index_name:
                self.index = None
            
            logger.info(f"Deleted index: {index_name}")
            
            return {
                "success": True,
                "message": f"Index '{index_name}' deleted successfully"
            }
            
        except Exception as e:
            logger.error(f"Error deleting index: {str(e)}")
            return {"error": f"Index deletion failed: {str(e)}"} 