import os
import numpy as np
from dotenv import load_dotenv
from embedding_system import EmbeddingSystem
import tempfile
import shutil


def test_api_key_validation():
    """Test 1: API Key Setup and Validation"""
    print("=== Test 1: API Key Setup and Validation ===")
    
    # Test with invalid API key
    try:
        system = EmbeddingSystem("invalid_key")
        result = system.generate_embedding("test text")
        if result is None:
            print("✓ Invalid API key correctly rejected")
        else:
            print("✗ Invalid API key should have failed")
    except Exception as e:
        print(f"✓ Invalid API key properly handled: {str(e)[:50]}...")
    
    # Test with valid API key
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        try:
            system = EmbeddingSystem(api_key)
            print("✓ Valid API key successfully initialized system")
            return system
        except Exception as e:
            print(f"✗ Valid API key failed: {e}")
            return None
    else:
        print("✗ No API key found in environment")
        return None


def test_embedding_generation(system):
    """Test 2: Embedding Model Response Testing"""
    print("\n=== Test 2: Embedding Model Response Testing ===")
    
    if not system:
        print("✗ Cannot test embedding generation without valid system")
        return None
    
    test_text = "This is a test sentence for embedding generation."
    
    try:
        embedding = system.generate_embedding(test_text)
        
        if embedding is not None:
            print(f"✓ Embedding generated successfully")
            print(f"✓ Embedding shape: {embedding.shape}")
            print(f"✓ Embedding dtype: {embedding.dtype}")
            print(f"✓ Expected dimension (3072): {len(embedding) == 3072}")
            
            # Test embedding properties
            if len(embedding) == 3072:
                print("✓ Embedding has correct dimension")
            else:
                print(f"✗ Expected 3072 dimensions, got {len(embedding)}")
            
            # Check if embedding contains valid numbers
            if np.all(np.isfinite(embedding)):
                print("✓ All embedding values are finite")
            else:
                print("✗ Embedding contains invalid values")
                
            return embedding
        else:
            print("✗ Embedding generation failed")
            return None
            
    except Exception as e:
        print(f"✗ Embedding generation error: {e}")
        return None


def test_vector_storage(system):
    """Test 3: Embedding Storage in Vector Database"""
    print("\n=== Test 3: Embedding Storage in Vector Database ===")
    
    if not system:
        print("✗ Cannot test vector storage without valid system")
        return False
    
    try:
        # Get initial stats
        initial_stats = system.get_stats()
        initial_count = initial_stats['total_documents']
        print(f"✓ Initial documents in system: {initial_count}")
        
        # Test adding a new text
        test_text = "This is a test document for vector storage validation."
        text_id = system.add_text(test_text)
        
        if text_id != -1:
            print(f"✓ Successfully added text with ID: {text_id}")
            
            # Verify storage increased
            new_stats = system.get_stats()
            if new_stats['total_documents'] == initial_count + 1:
                print("✓ Document count increased correctly")
            else:
                print("✗ Document count not updated properly")
            
            # Verify the text was stored
            if text_id in system.metadata and system.metadata[text_id]['text'] == test_text:
                print("✓ Text stored correctly in metadata")
            else:
                print("✗ Text not found in metadata")
            
            # Test persistence files exist
            if os.path.exists(system.index_path) and os.path.exists(system.metadata_path):
                print("✓ Persistence files exist")
            else:
                print("✗ Persistence files missing")
            
            return True
        else:
            print("✗ Failed to add test text")
            return False
        
    except Exception as e:
        print(f"✗ Vector storage test error: {e}")
        return False


def test_similarity_and_ranking(system):
    """Test 4: Cosine Similarity and Ranking"""
    print("\n=== Test 4: Cosine Similarity and Ranking ===")
    
    if not system:
        print("✗ Cannot test similarity without valid system")
        return False
    
    try:
        # Test similarity search with existing data
        query = "space exploration and astronauts"
        results = system.search_similar(query, k=3)
        
        if results:
            print(f"✓ Similarity search returned {len(results)} results")
            
            # Check ranking (distances should be ascending - lower is more similar)
            distances = [result[1] for result in results]
            if distances == sorted(distances):
                print("✓ Results are properly ranked by similarity (ascending distances)")
            else:
                print("✗ Results are not properly ranked")
            
            # Display results
            for i, (doc_id, distance, text) in enumerate(results, 1):
                similarity_score = 1 / (1 + distance)
                print(f"{i}. Score: {similarity_score:.3f}, Text: {text[:60]}...")
            
            # Test another query
            query2 = "artificial intelligence and emotions"
            results2 = system.search_similar(query2, k=2)
            
            if results2:
                print(f"\n✓ Second query returned {len(results2)} results")
                for i, (doc_id, distance, text) in enumerate(results2, 1):
                    similarity_score = 1 / (1 + distance)
                    print(f"{i}. Score: {similarity_score:.3f}, Text: {text[:60]}...")
            
            return True
        else:
            print("✗ No similarity results returned")
            return False
            
    except Exception as e:
        print(f"✗ Similarity test error: {e}")
        return False


def run_manual_similarity_test(system):
    """Manual similarity test with embeddings"""
    print("\n=== Manual Cosine Similarity Verification ===")
    
    if not system:
        print("✗ Cannot run manual test without valid system")
        return
    
    try:
        # Generate embeddings for similar texts
        text1 = "I love cats and dogs"
        text2 = "Pets like cats and dogs are wonderful"
        text3 = "The weather is sunny today"
        
        emb1 = system.generate_embedding(text1)
        emb2 = system.generate_embedding(text2)
        emb3 = system.generate_embedding(text3)
        
        if emb1 is not None and emb2 is not None and emb3 is not None:
            # Calculate manual cosine similarity
            def cosine_similarity(a, b):
                return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            
            sim_1_2 = cosine_similarity(emb1, emb2)
            sim_1_3 = cosine_similarity(emb1, emb3)
            
            print(f"✓ Similarity between similar texts: {sim_1_2:.3f}")
            print(f"✓ Similarity between different texts: {sim_1_3:.3f}")
            
            if sim_1_2 > sim_1_3:
                print("✓ Similar texts have higher similarity score")
            else:
                print("✗ Similarity scoring may need attention")
        
    except Exception as e:
        print(f"✗ Manual similarity test error: {e}")


def main():
    """Run all tests"""
    print("Starting Embedding System Unit Tests\n")
    
    # Test 1: API Key Validation
    system = test_api_key_validation()
    
    # Test 2: Embedding Generation
    embedding = test_embedding_generation(system)
    
    # Test 3: Vector Storage
    storage_success = test_vector_storage(system)
    
    # Test 4: Similarity and Ranking
    similarity_success = test_similarity_and_ranking(system)
    
    # Manual similarity verification
    run_manual_similarity_test(system)
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"API Key Validation: {'✓ PASS' if system else '✗ FAIL'}")
    print(f"Embedding Generation: {'✓ PASS' if embedding is not None else '✗ FAIL'}")
    print(f"Vector Storage: {'✓ PASS' if storage_success else '✗ FAIL'}")
    print(f"Similarity & Ranking: {'✓ PASS' if similarity_success else '✗ FAIL'}")
    
    all_passed = all([
        system is not None,
        embedding is not None,
        storage_success,
        similarity_success
    ])
    
    print(f"\nOverall Result: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")


if __name__ == "__main__":
    main()