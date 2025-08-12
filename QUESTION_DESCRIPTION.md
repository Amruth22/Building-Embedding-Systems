# Building Embedding Systems - Coding Challenge

## 🎯 Problem Statement

Build a complete **Text Embedding and Similarity Search System** using Google's Gemini AI model and FAISS vector database. Your task is to implement a production-ready system that can store text documents as high-dimensional embeddings and perform efficient similarity searches.

## 📋 Requirements

### Core Functionality
You need to implement an `EmbeddingSystem` class with the following capabilities:

1. **Embedding Generation**: Convert text into high-dimensional vectors using Google Gemini AI
2. **Vector Storage**: Store embeddings efficiently using FAISS vector database
3. **Similarity Search**: Find and rank similar documents based on semantic similarity
4. **Persistence**: Save and load embeddings and metadata to/from disk
5. **Error Handling**: Robust error management for API failures and edge cases

### Technical Specifications

#### Class: `EmbeddingSystem`

**Constructor Parameters:**
- `api_key` (str): Google Gemini API key
- `index_path` (str, optional): Path to save FAISS index (default: "faiss_index.bin")
- `metadata_path` (str, optional): Path to save metadata (default: "metadata.json")

**Required Methods:**

```python
def __init__(self, api_key: str, index_path: str = "faiss_index.bin", metadata_path: str = "metadata.json")
    # Initialize the embedding system

def generate_embedding(self, text: str) -> np.ndarray
    # Generate embedding for given text
    # Returns: numpy array of shape (3072,) or None if failed

def add_text(self, text: str) -> int
    # Add text to the system and return assigned ID
    # Returns: text_id (int) or -1 if failed

def search_similar(self, query_text: str, k: int = 5) -> List[Tuple[int, float, str]]
    # Search for similar texts
    # Returns: List of (text_id, distance, text) tuples

def get_stats(self) -> Dict
    # Return system statistics
    # Returns: {"total_documents": int, "index_size": int, "dimension": int}

def delete_all(self)
    # Clear all data and create fresh index
```

### Model Configuration
- **Model**: `gemini-embedding-001`
- **Embedding Dimension**: 3072
- **Task Type**: `RETRIEVAL_DOCUMENT`
- **Distance Metric**: L2 (Euclidean) distance

## 🧪 Test Cases

Your implementation will be tested against the following scenarios:

### Test Case 1: API Key Validation
```python
# Should handle invalid API keys gracefully
system = EmbeddingSystem("invalid_key")
result = system.generate_embedding("test")
assert result is None
```

### Test Case 2: Embedding Generation
```python
# Should generate valid embeddings
system = EmbeddingSystem(valid_api_key)
embedding = system.generate_embedding("Hello world")
assert embedding is not None
assert embedding.shape == (3072,)
assert embedding.dtype == np.float32
```

### Test Case 3: Document Storage
```python
# Should store documents and increment counter
initial_stats = system.get_stats()
text_id = system.add_text("Sample document")
new_stats = system.get_stats()
assert text_id >= 0
assert new_stats["total_documents"] == initial_stats["total_documents"] + 1
```

### Test Case 4: Similarity Search
```python
# Should return ranked similar documents
system.add_text("I love cats and dogs")
system.add_text("Pets are wonderful companions")
system.add_text("The weather is sunny today")

results = system.search_similar("Animals and pets", k=2)
assert len(results) == 2
assert all(len(result) == 3 for result in results)  # (id, distance, text)
# Results should be ranked by similarity (ascending distances)
assert results[0][1] <= results[1][1]
```

### Test Case 5: Persistence
```python
# Should save and load data correctly
system.add_text("Persistent document")
system._save_index()

# Create new system instance
new_system = EmbeddingSystem(api_key, same_paths)
assert new_system.get_stats()["total_documents"] > 0
```

### Test Case 6: Error Handling
```python
# Should handle various error conditions
assert system.add_text("") == -1  # Empty text
assert system.search_similar("", k=5) == []  # Empty query
```

## 📊 Evaluation Criteria

Your solution will be evaluated on:

1. **Correctness** (40%): All test cases pass
2. **Code Quality** (25%): Clean, readable, well-structured code
3. **Error Handling** (20%): Robust exception management
4. **Performance** (15%): Efficient implementation and resource usage

## 🔧 Implementation Guidelines

### Required Dependencies
```python
import os
import numpy as np
import faiss
import pickle
import json
from typing import List, Tuple, Dict, Optional
from google import genai
from google.genai.types import EmbedContentConfig
```

### Key Implementation Points

1. **FAISS Index**: Use `faiss.IndexFlatL2(3072)` for L2 distance-based similarity
2. **Embedding Config**: Configure with `task_type="RETRIEVAL_DOCUMENT"` and `output_dimensionality=3072`
3. **Persistence**: Save FAISS index as binary file and metadata as JSON
4. **Error Recovery**: Handle API failures, file I/O errors, and corrupted data
5. **Memory Management**: Ensure efficient handling of large embedding matrices

### Sample Usage
```python
# Initialize system
system = EmbeddingSystem(api_key="your_gemini_api_key")

# Add documents
doc_id = system.add_text("The quick brown fox jumps over the lazy dog")

# Search for similar content
results = system.search_similar("Fast animals jumping", k=3)
for doc_id, distance, text in results:
    similarity_score = 1 / (1 + distance)
    print(f"Similarity: {similarity_score:.3f} - {text}")

# Get system statistics
stats = system.get_stats()
print(f"Total documents: {stats['total_documents']}")
```

## 🚀 Bonus Features (Optional)

Implement these for extra credit:

1. **Batch Processing**: Add multiple documents at once
2. **Document Updates**: Update existing documents
3. **Metadata Filtering**: Search with additional filters
4. **Performance Monitoring**: Track API usage and response times
5. **Configuration Management**: Support for different models and parameters

## 📝 Submission Requirements

1. **Main Implementation**: `embedding_system.py` with the complete `EmbeddingSystem` class
2. **Example Usage**: `example.py` demonstrating system capabilities
3. **Test Suite**: `test_embedding_system.py` with comprehensive unit tests
4. **Environment Setup**: `.env` file template for API key configuration
5. **Documentation**: Clear comments and docstrings

## ⚠️ Important Notes

- **API Key**: You'll need a valid Google Gemini API key for testing
- **Dependencies**: Ensure all required packages are installed
- **File Permissions**: Your code should handle file I/O permissions gracefully
- **Memory Usage**: Be mindful of memory consumption with large document collections
- **Rate Limits**: Implement appropriate handling for API rate limits

## 🎯 Success Criteria

Your implementation is successful when:
- ✅ All unit tests pass
- ✅ System can handle 100+ documents efficiently
- ✅ Similarity search returns relevant results
- ✅ Data persists correctly across system restarts
- ✅ Error conditions are handled gracefully
- ✅ Code follows Python best practices

Good luck building your embedding system! 🚀