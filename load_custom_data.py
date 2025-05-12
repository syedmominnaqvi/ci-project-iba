#!/usr/bin/env python
"""
Example script for loading and analyzing custom PII/PHI datasets.
"""
import os
import sys
import argparse
import json
import pandas as pd
from src.genetic_algorithm import GeneticPIIDetector


def load_from_csv(file_path, text_column, annotations_column=None, delimiter=','):
    """
    Load text and annotations from a CSV file.
    
    Args:
        file_path: Path to CSV file
        text_column: Column name for text data
        annotations_column: Column name for annotations (if any)
        delimiter: CSV delimiter
        
    Returns:
        Tuple of (texts, annotations)
    """
    df = pd.read_csv(file_path, delimiter=delimiter)
    
    texts = df[text_column].tolist()
    
    # If annotations are provided in the CSV
    if annotations_column and annotations_column in df.columns:
        try:
            # Try to parse JSON annotations
            annotations = [json.loads(ann) if isinstance(ann, str) else [] 
                          for ann in df[annotations_column]]
        except:
            print(f"Warning: Could not parse annotations from column {annotations_column}")
            annotations = [[] for _ in texts]
    else:
        # No annotations
        annotations = [[] for _ in texts]
    
    return texts, annotations


def load_from_directory(dir_path, text_ext='.txt', ann_ext='.ann'):
    """
    Load text and annotations from a directory of text files.
    
    Args:
        dir_path: Path to directory with text files
        text_ext: File extension for text files
        ann_ext: File extension for annotation files
        
    Returns:
        Tuple of (texts, annotations)
    """
    texts = []
    annotations = []
    
    # Get all text files
    text_files = [f for f in os.listdir(dir_path) if f.endswith(text_ext)]
    
    for text_file in text_files:
        # Load text
        with open(os.path.join(dir_path, text_file), 'r') as f:
            text = f.read()
        texts.append(text)
        
        # Check for annotation file
        base_name = text_file[:-len(text_ext)]
        ann_file = base_name + ann_ext
        ann_path = os.path.join(dir_path, ann_file)
        
        if os.path.exists(ann_path):
            try:
                with open(ann_path, 'r') as f:
                    # Try to load as JSON
                    try:
                        anns = json.load(f)
                    except:
                        # Try to parse custom annotation format
                        # This is a placeholder - modify for your annotation format
                        anns = []
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) >= 3:
                                # Format: id, type start end, text
                                type_pos = parts[1].split(' ')
                                if len(type_pos) >= 3:
                                    ann_type = type_pos[0]
                                    start = int(type_pos[1])
                                    end = int(type_pos[2])
                                    anns.append((start, end, ann_type))
                
                annotations.append(anns)
            except Exception as e:
                print(f"Warning: Error parsing annotation file {ann_file}: {e}")
                annotations.append([])
        else:
            # No annotation file
            annotations.append([])
    
    return texts, annotations


def main():
    """Load custom data and analyze with the genetic algorithm."""
    parser = argparse.ArgumentParser(description="Load and analyze custom PII/PHI datasets")
    parser.add_argument("--input", type=str, required=True, 
                      help="Input data source (CSV file or directory path)")
    parser.add_argument("--input_type", type=str, choices=['csv', 'directory'], default='directory',
                      help="Type of input data source")
    parser.add_argument("--text_column", type=str, default='text',
                      help="Column name for text data in CSV")
    parser.add_argument("--ann_column", type=str, default=None,
                      help="Column name for annotations in CSV")
    parser.add_argument("--text_ext", type=str, default='.txt',
                      help="File extension for text files in directory")
    parser.add_argument("--ann_ext", type=str, default='.ann',
                      help="File extension for annotation files in directory")
    parser.add_argument("--generations", type=int, default=20,
                      help="Number of generations to train")
    parser.add_argument("--test_split", type=float, default=0.2,
                      help="Proportion of data to use for testing")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    
    if args.input_type == 'csv':
        texts, annotations = load_from_csv(
            args.input, 
            args.text_column, 
            args.ann_column
        )
    else:  # directory
        texts, annotations = load_from_directory(
            args.input,
            args.text_ext,
            args.ann_ext
        )
    
    print(f"Loaded {len(texts)} documents with {sum(len(a) for a in annotations)} annotations")
    
    # Split into train/test
    import random
    from sklearn.model_selection import train_test_split
    
    train_texts, test_texts, train_anns, test_anns = train_test_split(
        texts, annotations, test_size=args.test_split, random_state=42
    )
    
    print(f"Training on {len(train_texts)} documents, testing on {len(test_texts)} documents")
    
    # Train genetic algorithm
    print(f"Training genetic algorithm for {args.generations} generations...")
    
    genetic_model = GeneticPIIDetector(
        population_size=50,
        generations=args.generations,
        crossover_prob=0.7,
        mutation_prob=0.2
    )
    
    genetic_model.train(train_texts, train_anns)
    
    # Evaluate
    print("\nEvaluating on test set...")
    metrics = genetic_model.evaluate(test_texts, test_anns)
    
    print("\nResults:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    
    # Sample predictions
    print("\nSample predictions on test documents:")
    
    for i, text in enumerate(test_texts[:3]):  # Show first 3 test docs
        print(f"\nTest Document {i+1}:")
        print(text[:300] + "..." if len(text) > 300 else text)
        
        predictions = genetic_model.predict(text)
        
        print("\nDetected entities:")
        for match, start, end, pii_type, confidence in predictions:
            print(f"- {pii_type}: '{match}' (confidence: {confidence:.2f})")
    
    print("\nTraining complete. The model is now ready for predictions.")
    
    # Allow interactive testing
    while True:
        try:
            print("\nEnter text to analyze (or 'quit' to exit):")
            user_text = input("> ")
            
            if user_text.lower() in ['quit', 'exit', 'q']:
                break
            
            predictions = genetic_model.predict(user_text)
            
            print("\nDetected entities:")
            if not predictions:
                print("No PII/PHI detected")
            else:
                for match, start, end, pii_type, confidence in predictions:
                    print(f"- {pii_type}: '{match}' (confidence: {confidence:.2f})")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()