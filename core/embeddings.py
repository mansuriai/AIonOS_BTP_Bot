# core/embeddings.py
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from utils.config import config
from typing import List

class EmbeddingManager:
    def __init__(self): 
        print(f"Loading model from: {config.EMBEDDING_MODEL}")
        # Make sure we're using the correct model with the right dimensions
        self.model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
        )
        
        # Add a dimension check during initialization
        test_embedding = self.model.embed_documents(["Test dimension check"])
        if test_embedding and len(test_embedding[0]) != config.EMBEDDING_DIMENSION:
            raise ValueError(f"Model produces embeddings with dimension {len(test_embedding[0])}, but config.EMBEDDING_DIMENSION is set to {config.EMBEDDING_DIMENSION}")
    
    # def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
    #     """Generate embeddings for a list of texts."""
    #     embeddings = self.model.embed_documents(texts)
        
    #     # Double-check dimensions before returning
    #     if embeddings and len(embeddings[0]) != config.EMBEDDING_DIMENSION:
    #         raise ValueError(f"Generated embeddings have dimension {len(embeddings[0])}, but expected {config.EMBEDDING_DIMENSION}")
            
    #     return embeddings

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Add input validation
        if not texts:
            return []
        
        # Truncate extremely long texts to prevent processing issues
        max_text_length = 2000  # Adjust based on model capabilities
        truncated_texts = [
            text[:max_text_length] if len(text) > max_text_length else text 
            for text in texts
        ]
        
        embeddings = self.model.embed_documents(truncated_texts)
        
        # Additional dimension check
        if any(len(emb) != config.EMBEDDING_DIMENSION for emb in embeddings):
            raise ValueError(f"Embedding dimension mismatch. Expected {config.EMBEDDING_DIMENSION}")
        
        return embeddings