import os
import numpy as np
import faiss
import pickle
import json
from typing import List, Tuple, Dict, Optional
from google import genai
from google.genai.types import EmbedContentConfig


class EmbeddingSystem:
    def __init__(self, api_key: str, index_path: str = "faiss_index.bin", metadata_path: str = "metadata.json"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-embedding-001"
        self.dimension = 3072
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        self.index = None
        self.metadata = {}
        self.text_counter = 0
        
        self._initialize_index()
    
    def _initialize_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self._load_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = {}
        self.text_counter = 0
        print("Created new FAISS index")
    
    def _load_index(self):
        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            self.text_counter = len(self.metadata)
            print(f"Loaded existing index with {self.text_counter} documents")
        except Exception as e:
            print(f"Error loading index: {e}")
            self._create_new_index()
    
    def _save_index(self):
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            print("Index saved successfully")
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=[text],
                config=EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=self.dimension,
                    title="Document Embedding"
                )
            )
            
            # Handle different response structures
            if hasattr(response, 'embedding'):
                embedding = np.array(response.embedding.values, dtype=np.float32)
            elif hasattr(response, 'embeddings') and len(response.embeddings) > 0:
                embedding = np.array(response.embeddings[0].values, dtype=np.float32)
            else:
                print(f"Unexpected response structure: {response}")
                return None
            
            return embedding
        
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def add_text(self, text: str) -> int:
        embedding = self.generate_embedding(text)
        if embedding is None:
            return -1
        
        text_id = self.text_counter
        
        embedding = embedding.reshape(1, -1)
        self.index.add(embedding)
        
        self.metadata[text_id] = {
            "text": text
        }
        
        self.text_counter += 1
        self._save_index()
        
        print(f"Added text with ID: {text_id}")
        return text_id
    
    def search_similar(self, query_text: str, k: int = 5) -> List[Tuple[int, float, str]]:
        query_embedding = self.generate_embedding(query_text)
        if query_embedding is None:
            return []
        
        query_embedding = query_embedding.reshape(1, -1)
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:
                continue
            
            text_data = self.metadata.get(idx, {})
            results.append((
                idx,
                float(distance),
                text_data.get("text", "")
            ))
        
        return results
    
    def get_stats(self) -> Dict:
        return {
            "total_documents": self.text_counter,
            "index_size": self.index.ntotal if self.index else 0,
            "dimension": self.dimension
        }
    
    def delete_all(self):
        self._create_new_index()
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
        print("Deleted all data and created fresh index")