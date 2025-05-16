#!/usr/bin/env python
"""
Simple PHI detector comparison - no external libraries required.
Only compares the genetic algorithm with a regex baseline.
"""
import os
import sys
import json
import glob
import random
from datetime import datetime

# Import our PHI detection system
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.genetic_algorithm import GeneticPIIDetector
from src.chromosome import DetectionGene, Chromosome


def create_regex_baseline():
    """Create a simple regex-based baseline model."""
    genes = [
        DetectionGene(pattern=r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', pii_type="NAME"),
        DetectionGene(pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', pii_type="EMAIL"),
        DetectionGene(pattern=r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', pii_type="PHONE"),
        DetectionGene(pattern=r'\b\d{3}[-]?\d{2}[-]?\d{4}\b', pii_type="SSN"),
        DetectionGene(pattern=r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', pii_type="DATE"),
        DetectionGene(pattern=r'\b\d+\s+[A-Z][a-z]+\s+[A-Za-z]+\b', pii_type="ADDRESS"),
    ]
    return Chromosome(genes=genes)


def find_text_files(directory):
    """Find all text files in a directory."""
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return []
    
    text_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                text_files.append(os.path.join(root, file))
    
    return text_files


def read_file(file_path):
    """Read a text file safely."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def main():
    """Run a simple comparison between genetic algorithm and regex baseline."""
    # Parse arguments
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "medical_samples"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "simple_comparison.json"
    generations = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    max_files = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    
    # Find text files
    print(f"Finding text files in {input_dir}...")
    file_paths = find_text_files(input_dir)
    
    if not file_paths:
        print("No text files found!")
        return 1
    
    print(f"Found {len(file_paths)} text files")
    
    # Limit number of files if needed
    if max_files and len(file_paths) > max_files:
        print(f"Limiting to {max_files} random files")
        file_paths = random.sample(file_paths, max_files)
    
    # Read files
    print("Reading files...")
    samples = []
    for file_path in file_paths:
        text = read_file(file_path)
        if text:
            samples.append((file_path, text))
    
    if not samples:
        print("No valid text content found!")
        return 1
    
    print(f"Successfully read {len(samples)} files")
    
    # Create models
    print(f"Training genetic algorithm for {generations} generations...")
    genetic_model = GeneticPIIDetector(generations=generations)
    
    # Train with empty annotations (unsupervised)
    genetic_model.train([text for _, text in samples], [[] for _ in samples])
    
    print("Creating regex baseline...")
    regex_baseline = create_regex_baseline()
    
    # Process each sample
    results = {
        "genetic": [],
        "regex": []
    }
    
    print("Running detection...")
    for file_path, text in samples:
        file_name = os.path.basename(file_path)
        print(f"Processing {file_name}...")
        
        # Genetic algorithm detection
        genetic_preds = genetic_model.predict(text)
        
        # Regex baseline detection
        regex_preds = regex_baseline.detect(text)
        
        # Record results
        results["genetic"].append({
            "file": file_name,
            "file_path": file_path,
            "count": len(genetic_preds),
            "types": list(set(p[3] for p in genetic_preds)),
            "entities": [{"text": p[0], "type": p[3], "confidence": float(p[4])} for p in genetic_preds]
        })
        
        results["regex"].append({
            "file": file_name,
            "file_path": file_path,
            "count": len(regex_preds),
            "types": list(set(p[3] for p in regex_preds)),
            "entities": [{"text": p[0], "type": p[3], "confidence": float(p[4])} for p in regex_preds]
        })
    
    # Calculate summary
    summary = {}
    
    # Genetic model stats
    genetic_total = sum(r["count"] for r in results["genetic"])
    genetic_types = set()
    for r in results["genetic"]:
        genetic_types.update(r["types"])
    
    summary["genetic"] = {
        "total_entities": genetic_total,
        "avg_per_file": genetic_total / len(samples) if samples else 0,
        "entity_types": list(genetic_types)
    }
    
    # Regex model stats
    regex_total = sum(r["count"] for r in results["regex"])
    regex_types = set()
    for r in results["regex"]:
        regex_types.update(r["types"])
    
    summary["regex"] = {
        "total_entities": regex_total,
        "avg_per_file": regex_total / len(samples) if samples else 0,
        "entity_types": list(regex_types)
    }
    
    # Calculate overlap
    overlap_count = 0
    only_genetic = 0
    only_regex = 0
    
    for i in range(len(samples)):
        genetic_entities = {e["text"] for e in results["genetic"][i]["entities"]}
        regex_entities = {e["text"] for e in results["regex"][i]["entities"]}
        
        overlap = genetic_entities.intersection(regex_entities)
        overlap_count += len(overlap)
        only_genetic += len(genetic_entities - regex_entities)
        only_regex += len(regex_entities - genetic_entities)
    
    summary["overlap"] = {
        "both_models": overlap_count,
        "only_genetic": only_genetic,
        "only_regex": only_regex
    }
    
    # Save results
    output = {
        "metadata": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "files_processed": len(samples),
            "genetic_generations": generations
        },
        "summary": summary,
        "detailed_results": results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        # Use a function to make complex objects JSON serializable
        json.dump(output, f, indent=2, default=str)
    
    # Print summary
    print("\nCOMPARISON SUMMARY:")
    print(f"\nGenetic Algorithm:")
    print(f"  Total entities: {summary['genetic']['total_entities']}")
    print(f"  Avg per file: {summary['genetic']['avg_per_file']:.2f}")
    print(f"  Entity types: {', '.join(summary['genetic']['entity_types'])}")
    
    print(f"\nRegex Baseline:")
    print(f"  Total entities: {summary['regex']['total_entities']}")
    print(f"  Avg per file: {summary['regex']['avg_per_file']:.2f}")
    print(f"  Entity types: {', '.join(summary['regex']['entity_types'])}")
    
    print(f"\nOverlap Analysis:")
    print(f"  Detected by both models: {summary['overlap']['both_models']}")
    print(f"  Only by genetic algorithm: {summary['overlap']['only_genetic']}")
    print(f"  Only by regex baseline: {summary['overlap']['only_regex']}")
    
    print(f"\nResults saved to {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())