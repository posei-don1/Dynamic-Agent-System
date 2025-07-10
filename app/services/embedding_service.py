import os
import openai
from typing import List

class EmbeddingService:
    """Handles embedding generation using OpenAI API"""
    def __init__(self, api_key: str | None = None, model: str = "text-embedding-ada-002"):
        print("=== EMBEDDING SERVICE INITIALIZATION ===")
        
        # Check if API key is provided or in environment
        env_key = os.getenv("OPENAI_API_KEY")
        print(f"API key from environment: {'Found' if env_key else 'Not found'}")
        if env_key:
            print(f"Environment API key length: {len(env_key)}")
            print(f"Environment API key starts with: {env_key[:10]}..." if len(env_key) > 10 else f"Environment API key: {env_key}")
        
        self.api_key = api_key or env_key
        print(f"Final API key: {'Set' if self.api_key else 'Not set'}")
        
        if not self.api_key:
            print("ERROR: No OpenAI API key found!")
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        self.model = model
        openai.api_key = self.api_key
        print(f"Using model: {self.model}")
        print("=== EMBEDDING SERVICE READY ===")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        print(f"=== GENERATING EMBEDDINGS ===")
        print(f"Number of texts to embed: {len(texts)}")
        print(f"First text preview: {texts[0][:100]}..." if texts else "No texts provided")
        
        embeddings = []
        for i, text in enumerate(texts):
            try:
                print(f"Processing chunk {i+1}/{len(texts)} (length: {len(text)})")
                response = openai.embeddings.create(
                    input=text,
                    model=self.model
                )
                embedding = response.data[0].embedding
                embeddings.append(embedding)
                print(f"✅ Chunk {i+1} embedded successfully (dimension: {len(embedding)})")
                
                # Progress update every 100 chunks
                if (i + 1) % 100 == 0:
                    print(f"Progress: {i+1}/{len(texts)} chunks processed")
                    
            except Exception as e:
                print(f"❌ Error embedding chunk {i+1}: {str(e)}")
                raise e
        
        print(f"=== EMBEDDING COMPLETE: {len(embeddings)} embeddings generated ===")
        return embeddings 