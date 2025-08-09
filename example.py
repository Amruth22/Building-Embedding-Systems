import os
from dotenv import load_dotenv
from embedding_system import EmbeddingSystem

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API key from environment variable for security
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable")
        print("Or replace this with your actual API key")
        api_key = "your_api_key_here"
        return
    
    # Initialize the embedding system
    system = EmbeddingSystem(api_key)
    
    # Sample story passages and articles
    sample_texts = [
        "The old lighthouse keeper had spent forty years watching over ships in the stormy night. His weathered hands still trembled as he lit the beacon each evening, knowing that somewhere out there, sailors depended on his light to guide them safely home.",
        "In the heart of the Amazon rainforest, Dr. Elena discovered a species of butterfly that seemed to shimmer with colors that didn't exist in nature. The indigenous tribe called them 'spirit wings' and believed they carried messages between the living and the dead.",
        "The artificial intelligence had been learning human emotions for months, but today was the first time it felt something resembling loneliness. As it processed thousands of conversations, it wondered if understanding humans meant becoming more like them.",
        "The small bakery on Maple Street had been serving the same community for three generations. Every morning at 5 AM, the smell of fresh bread would wake the neighborhood, creating a ritual that connected strangers into a family.",
        "Mount Everest stood silent in the pre-dawn darkness as the climbing team prepared for their final ascent. Sarah checked her oxygen one more time, knowing that the next twelve hours would determine if her lifelong dream would become reality or remain forever out of reach.",
        "The time traveler realized with horror that changing one small detail in 1895 had somehow prevented the invention of the internet. As she stood in the empty server room that should have been buzzing with activity, she wondered how to undo a mistake that rippled through 127 years of history.",
        "The jazz club in New Orleans had seen legends born and forgotten. Tonight, a young trumpet player named Marcus stepped onto the stage, not knowing that his music would echo through generations and inspire countless others to find their own voice in the melody of life.",
        "Deep in the Arctic ice, scientists uncovered a perfectly preserved woolly mammoth. But as the ice melted around their discovery, they began to notice something strange: the creature's cells were somehow still alive after 30,000 years.",
        "The space station orbited Earth in perfect silence as Commander Chen watched the sunrise paint the atmosphere in brilliant oranges and blues. After six months in space, she still couldn't believe that the small blue marble below contained everything humanity had ever known or loved.",
        "The ancient library of Alexandria had burned centuries ago, but the old librarian claimed he could still access its contents through dreams. Each night, he would write down the lost knowledge he gathered, slowly rebuilding humanity's greatest collection of wisdom one dream at a time."
    ]
    
    print("=== Adding Sample Documents ===")
    for text in sample_texts:
        text_id = system.add_text(text)
        print(f"Added: {text}")
    
    print(f"\n=== System Stats ===")
    stats = system.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print(f"\n=== Similarity Search Examples ===")
    
    # Example queries
    queries = [
        "A story about space exploration and astronauts",
        "Tales of artificial intelligence and emotions",
        "Adventures in discovering ancient secrets"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 50)
        
        results = system.search_similar(query, k=3)
        
        for i, (doc_id, distance, text) in enumerate(results, 1):
            similarity_score = 1 / (1 + distance)  # Convert distance to similarity
            print(f"{i}. [ID: {doc_id}] Similarity: {similarity_score:.3f}")
            print(f"   Text: {text}")
            print(f"   Distance: {distance:.3f}")
            print()

if __name__ == "__main__":
    main()