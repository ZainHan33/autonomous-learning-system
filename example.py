import asyncio
from learning_manager import LearningManager
import textwrap
from content_processor import ContentProcessor
from visualizer import ConceptVisualizer

async def main():
    # Initialize managers
    learner = LearningManager()
    visualizer = ConceptVisualizer()
    
    # Topic to learn about
    topic = "quantum computing"
    
    print(f"\nLearning about {topic}...")
    print("This may take a few moments due to rate limiting...")
    
    results = await learner.learn_topic(topic, depth=2)
    
    print(f"\nFound {len(results)} relevant pieces of information")
    
    # Track unique concepts and relations
    all_concepts = set()
    all_relations = []
    
    # Process and display results
    for i, result in enumerate(results, 1):
        print(f"\n{'-'*80}")
        print(f"{i}. Topic: {result.topic}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Source: {result.source}\n")
        
        # Process the content
        processed = learner.content_processor.process_content(result.content)
        
        # Main Concepts
        if processed.main_concepts:
            print("Main Concepts:")
            for concept in processed.main_concepts:
                print(f"- {concept}")
                all_concepts.add(concept)
        
        # Key Points
        if processed.key_points:
            print("\nKey Points:")
            for point in processed.key_points:
                wrapped_point = textwrap.fill(point, width=80, initial_indent="- ", 
                                          subsequent_indent="  ")
                print(wrapped_point)
        
        # Definitions
        if processed.definitions:
            print("\nDefinitions Found:")
            for term, definition in processed.definitions.items():
                wrapped_def = textwrap.fill(f"{term}: {definition}", width=80, 
                                        initial_indent="- ", subsequent_indent="  ")
                print(wrapped_def)
        
        # Relations
        if processed.relations:
            print("\nRelationships Found:")
            for subj, rel, obj in processed.relations:
                print(f"- {subj} {rel} {obj}")
                all_relations.append((subj, rel, obj))
    
    # Create visualization
    print("\nGenerating concept visualization...")
    visualizer.visualize_concepts(
        concepts=all_concepts,
        relations=all_relations,
        title=f"Concept Map: {topic}",
        filename="quantum_computing_concepts.svg"
    )
    print("Visualization saved as 'quantum_computing_concepts.svg'")
    
    # Summary section
    print(f"\n{'-'*80}")
    print("\nSummary:")
    print(f"Total unique concepts discovered: {len(all_concepts)}")
    print(f"Total relationships mapped: {len(all_relations)}")
    
    print("\nKey Concept Map:")
    concepts_by_frequency = {}
    for concept in all_concepts:
        freq = sum(1 for rel in all_relations if concept in (rel[0], rel[2]))
        if freq > 0:
            concepts_by_frequency[concept] = freq
    
    for concept, freq in sorted(concepts_by_frequency.items(), key=lambda x: x[1], reverse=True):
        print(f"- {concept} (connected to {freq} other concepts)")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nLearning process interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
