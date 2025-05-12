#!/usr/bin/env python
"""
Visualize PII/PHI detection on a document.
"""
import os
import sys
import argparse
import json
from src.data_utils import PIIDataset
from src.genetic_algorithm import GeneticPIIDetector
from src.baseline import PresidioBaseline
from colorama import init, Fore, Back, Style

# Initialize colorama
init()

def highlight_entities(text, entities):
    """
    Highlights entities in the text with color-coding by type.
    
    Args:
        text: String text to highlight
        entities: List of (match, start, end, pii_type, confidence) tuples
        
    Returns:
        String with color-coded highlighting
    """
    # Sort entities by start position, reversed (to avoid messing up positions)
    sorted_entities = sorted(entities, key=lambda x: x[1], reverse=True)
    
    # Color mapping for different entity types
    color_map = {
        "NAME": Fore.RED,
        "EMAIL": Fore.GREEN,
        "PHONE": Fore.YELLOW,
        "SSN": Fore.MAGENTA,
        "ADDRESS": Fore.BLUE,
        "MEDICAL_RECORD": Fore.CYAN,
        "DIAGNOSIS": Fore.WHITE + Back.RED,
        "MEDICATION": Fore.BLACK + Back.GREEN,
        "OTHER": Fore.WHITE
    }
    
    # Apply highlighting
    result = text
    for match, start, end, pii_type, confidence in sorted_entities:
        color = color_map.get(pii_type, Fore.WHITE)
        # Replace with colored version
        highlighted = f"{color}[{pii_type}: {match}]{Style.RESET_ALL}"
        result = result[:start] + highlighted + result[end:]
    
    return result

def main():
    """Visualize PII/PHI detection."""
    parser = argparse.ArgumentParser(description="Visualize PII/PHI detection")
    parser.add_argument("--input_file", type=str, help="Path to input text file")
    parser.add_argument("--model", type=str, choices=["genetic", "presidio", "both"], 
                       default="both", help="Model to use for detection")
    parser.add_argument("--train_size", type=int, default=20, 
                       help="Number of documents to train genetic model on")
    parser.add_argument("--generations", type=int, default=10,
                       help="Number of generations for genetic model")
    
    args = parser.parse_args()
    
    # Get input text
    if args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Error: File {args.input_file} not found")
            sys.exit(1)
        
        with open(args.input_file, "r") as f:
            text = f.read()
    else:
        # Generate a random document
        print("No input file provided. Generating a random document...")
        dataset = PIIDataset()
        text, annotations = dataset.generate_synthetic_document()
        print(f"Ground truth annotations: {len(annotations)} entities\n")
    
    # Initialize models
    models = {}
    
    if args.model in ["genetic", "both"]:
        print(f"Training genetic model on {args.train_size} documents for {args.generations} generations...")
        # Generate some training data
        dataset = PIIDataset()
        train_data = dataset.generate_dataset(num_documents=args.train_size)
        
        # Train genetic model
        genetic_model = GeneticPIIDetector(
            population_size=30,
            generations=args.generations,
            crossover_prob=0.7,
            mutation_prob=0.2
        )
        
        genetic_model.train(
            train_data["train"]["texts"],
            train_data["train"]["annotations"]
        )
        models["Genetic Algorithm"] = genetic_model.predict
    
    if args.model in ["presidio", "both"]:
        print("Initializing Presidio baseline...")
        try:
            baseline_model = PresidioBaseline()
            models["Presidio Baseline"] = baseline_model.detect
        except Exception as e:
            print(f"Error initializing Presidio: {e}")
            print("Using a simple regex baseline instead.")
            from src.chromosome import DetectionGene, Chromosome
            
            # Create a simple rule-based baseline
            genes = [
                DetectionGene(pattern=r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', pii_type="NAME"),
                DetectionGene(pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', pii_type="EMAIL"),
                DetectionGene(pattern=r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', pii_type="PHONE"),
                DetectionGene(pattern=r'\b\d{3}[-]?\d{2}[-]?\d{4}\b', pii_type="SSN"),
                DetectionGene(pattern=r'\b\d+\s+[A-Z][a-z]+\s+[A-Za-z]+\b', pii_type="ADDRESS"),
                DetectionGene(pattern=r'\bMRN[-:]?\s*\d+\b', pii_type="MEDICAL_RECORD"),
            ]
            baseline_chromosome = Chromosome(genes=genes)
            
            models["Simple Regex Baseline"] = baseline_chromosome.detect
    
    # Print original text
    print("\nORIGINAL TEXT:")
    print("=" * 80)
    print(text)
    print("=" * 80)
    
    # Run detection with each model
    print("\nDETECTION RESULTS:\n")
    
    for model_name, detect_func in models.items():
        print(f"\n{model_name}:")
        print("-" * 80)
        
        # Get entities
        entities = detect_func(text)
        
        # Print highlighted text
        highlighted_text = highlight_entities(text, entities)
        print(highlighted_text)
        
        # Print entity summary
        print("\nDetected entities:")
        for match, start, end, pii_type, confidence in sorted(entities, key=lambda x: x[1]):
            print(f"- {pii_type}: '{match}' (confidence: {confidence:.2f})")
        
        print(f"\nTotal: {len(entities)} entities detected\n")

if __name__ == "__main__":
    main()