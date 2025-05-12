#!/usr/bin/env python
"""
Demonstration script for the PII/PHI detection system.
"""
import os
import sys
import argparse
from src.data_utils import PIIDataset
from src.genetic_algorithm import GeneticPIIDetector
from src.baseline import PresidioBaseline


def compare_detection(text, genetic_detector, baseline_detector):
    """
    Compare detection results from different approaches.
    
    Args:
        text: Text to analyze
        genetic_detector: Trained genetic algorithm detector
        baseline_detector: Baseline detector
    """
    print("\n" + "="*80)
    print(f"TEXT: {text}")
    print("="*80)
    
    # Get predictions from each model
    genetic_preds = genetic_detector.predict(text)
    baseline_preds = baseline_detector.detect(text)
    
    # Print genetic algorithm results
    print("\nGENETIC ALGORITHM RESULTS:")
    print("-"*40)
    if not genetic_preds:
        print("No PII/PHI detected")
    else:
        for match, start, end, pii_type, confidence in genetic_preds:
            print(f"[{pii_type}] '{match}' (confidence: {confidence:.2f})")
    
    # Print baseline results
    print("\nBASELINE RESULTS:")
    print("-"*40)
    if not baseline_preds:
        print("No PII/PHI detected")
    else:
        for match, start, end, pii_type, confidence in baseline_preds:
            print(f"[{pii_type}] '{match}' (confidence: {confidence:.2f})")
    
    print("\n")


def main():
    """Run the demonstration."""
    parser = argparse.ArgumentParser(description="PII/PHI Detection Demo")
    parser.add_argument("--train_size", type=int, default=20, 
                       help="Number of documents to train on")
    parser.add_argument("--generations", type=int, default=10,
                       help="Number of generations to evolve")
    parser.add_argument("--custom_text", type=str, default=None,
                       help="Custom text to analyze (if not provided, will generate examples)")
    
    args = parser.parse_args()
    
    # Generate data
    print(f"Generating {args.train_size} training documents...")
    dataset = PIIDataset()
    train_data = dataset.generate_dataset(num_documents=args.train_size)
    
    # Train the genetic algorithm model
    print(f"Training genetic algorithm for {args.generations} generations...")
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
    
    # Initialize baseline
    print("Initializing baseline model...")
    try:
        baseline_model = PresidioBaseline()
    except Exception as e:
        print(f"Error initializing Presidio: {e}")
        print("Using a simple regex baseline instead.")
        from src.chromosome import DetectionGene, Chromosome
        
        # Create a simple rule-based baseline using our chromosome system
        genes = [
            DetectionGene(pattern=r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', pii_type="NAME"),
            DetectionGene(pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', pii_type="EMAIL"),
            DetectionGene(pattern=r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', pii_type="PHONE"),
            DetectionGene(pattern=r'\b\d{3}[-]?\d{2}[-]?\d{4}\b', pii_type="SSN"),
            DetectionGene(pattern=r'\b\d+\s+[A-Z][a-z]+\s+[A-Za-z]+\b', pii_type="ADDRESS"),
            DetectionGene(pattern=r'\bMRN[-:]?\s*\d+\b', pii_type="MEDICAL_RECORD"),
        ]
        baseline_chromosome = Chromosome(genes=genes)
        
        class SimpleBaseline:
            def detect(self, text):
                return baseline_chromosome.detect(text)
        
        baseline_model = SimpleBaseline()
    
    # Test on custom text or generate examples
    if args.custom_text:
        compare_detection(args.custom_text, genetic_model, baseline_model)
    else:
        # Generate a few examples
        print("\nTesting on generated examples...")
        for i in range(3):
            text, _ = dataset.generate_synthetic_document()
            compare_detection(text, genetic_model, baseline_model)


if __name__ == "__main__":
    main()