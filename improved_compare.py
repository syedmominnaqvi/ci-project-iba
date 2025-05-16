#!/usr/bin/env python
"""
Improved PHI detector comparison.
Enhances the genetic algorithm with:
1. Proper seed patterns
2. More generations
3. Better debugging
4. Synthetic annotations for training
"""
import os
import sys
import json
import glob
import random
import re
import time
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


def create_seed_patterns():
    """Create better seed patterns for the genetic algorithm."""
    # Start with the baseline patterns but make more variations
    genes = [
        # Names
        DetectionGene(pattern=r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', pii_type="NAME", confidence=0.7),
        DetectionGene(pattern=r'\b[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+\b', pii_type="NAME", confidence=0.8),
        DetectionGene(pattern=r'\bDr\.\s+[A-Z][a-z]+\b', pii_type="NAME", confidence=0.8),
        DetectionGene(pattern=r'\b[A-Z][a-z]+,\s+[A-Z][a-z]+\b', pii_type="NAME", confidence=0.7),
        
        # Emails
        DetectionGene(pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', pii_type="EMAIL", confidence=0.9),
        
        # Phone numbers (various formats)
        DetectionGene(pattern=r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', pii_type="PHONE", confidence=0.8),
        DetectionGene(pattern=r'\(\d{3}\)\s*\d{3}[-.]?\d{4}\b', pii_type="PHONE", confidence=0.8),
        
        # SSN
        DetectionGene(pattern=r'\b\d{3}[-]?\d{2}[-]?\d{4}\b', pii_type="SSN", confidence=0.9),
        
        # Dates
        DetectionGene(pattern=r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', pii_type="DATE", confidence=0.7),
        DetectionGene(pattern=r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b', pii_type="DATE", confidence=0.8),
        DetectionGene(pattern=r'\b\d{1,2}-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-a-z]*-\d{2,4}\b', pii_type="DATE", confidence=0.8),
        
        # Addresses
        DetectionGene(pattern=r'\b\d+\s+[A-Z][a-z]+\s+[A-Za-z]+\b', pii_type="ADDRESS", confidence=0.7),
        DetectionGene(pattern=r'\b\d+\s+[A-Z][a-z]+\s+(St|Ave|Rd|Blvd|Drive|Lane|Court|Way)\b', pii_type="ADDRESS", confidence=0.8),
        
        # Medical record numbers
        DetectionGene(pattern=r'\bMRN:\s*\d+\b', pii_type="MEDICAL_RECORD", confidence=0.9),
        DetectionGene(pattern=r'\bPatient ID:\s*\d+\b', pii_type="MEDICAL_RECORD", confidence=0.9),
        DetectionGene(pattern=r'\bRecord #\s*\d+\b', pii_type="MEDICAL_RECORD", confidence=0.8),
        
        # Diagnosis codes
        DetectionGene(pattern=r'\bICD-10:\s*[A-Z]\d+\.\d+\b', pii_type="DIAGNOSIS", confidence=0.9),
        DetectionGene(pattern=r'\bDiagnosis Code:\s*[A-Z]\d+\.\d+\b', pii_type="DIAGNOSIS", confidence=0.9),
        
        # Medications
        DetectionGene(pattern=r'\b\d+\s*mg\s+[A-Z][a-z]+\b', pii_type="MEDICATION", confidence=0.8),
        DetectionGene(pattern=r'\b(daily|BID|TID|QID)\b', pii_type="MEDICATION", confidence=0.6),
    ]
    return genes


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


def generate_synthetic_annotations(text, regex_baseline):
    """
    Generate synthetic annotations based on regex patterns.
    This helps provide training examples for the genetic algorithm.
    """
    # Get detections from regex baseline
    detections = regex_baseline.detect(text)
    
    # Convert to annotations format (start, end, type)
    annotations = [(start, end, pii_type) for _, start, end, pii_type, _ in detections]
    
    # Add 20% random noise by removing some annotations
    if annotations:
        num_to_remove = max(1, int(len(annotations) * 0.2))
        indices_to_remove = random.sample(range(len(annotations)), num_to_remove)
        filtered_annotations = [ann for i, ann in enumerate(annotations) if i not in indices_to_remove]
        return filtered_annotations
    
    return annotations


def process_batch(samples, batch_idx, total_batches, genetic_model, regex_baseline):
    """Process a batch of samples to avoid memory issues."""
    print(f"\nProcessing batch {batch_idx}/{total_batches}...")
    
    batch_results = {
        "genetic": [],
        "regex": []
    }
    
    # Process each sample in the batch
    for i, (file_path, text) in enumerate(samples):
        file_name = os.path.basename(file_path)
        print(f"  Processing file {i+1}/{len(samples)}: {file_name}")
        
        # Track timing
        start_time = time.time()
        
        # Genetic algorithm detection
        print(f"    Running genetic algorithm...")
        genetic_preds = genetic_model.predict(text)
        genetic_time = time.time() - start_time
        
        # Regex baseline detection
        print(f"    Running regex baseline...")
        start_time = time.time()
        regex_preds = regex_baseline.detect(text)
        regex_time = time.time() - start_time
        
        # Print detection counts
        print(f"    Found {len(genetic_preds)} genetic matches, {len(regex_preds)} regex matches")
        
        # Record results
        batch_results["genetic"].append({
            "file": file_name,
            "file_path": file_path,
            "count": len(genetic_preds),
            "types": list(set(p[3] for p in genetic_preds)),
            "time_seconds": genetic_time,
            "entities": [{"text": p[0], "type": p[3], "confidence": float(p[4])} for p in genetic_preds]
        })
        
        batch_results["regex"].append({
            "file": file_name,
            "file_path": file_path,
            "count": len(regex_preds),
            "types": list(set(p[3] for p in regex_preds)),
            "time_seconds": regex_time,
            "entities": [{"text": p[0], "type": p[3], "confidence": float(p[4])} for p in regex_preds]
        })
    
    return batch_results


def main():
    """Run improved comparison between genetic algorithm and regex baseline."""
    # Parse arguments
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "improved_comparison.json"
    generations = int(sys.argv[3]) if len(sys.argv) > 3 else 50  # Increased from 10 to 50
    max_files = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    batch_size = int(sys.argv[5]) if len(sys.argv) > 5 else 5
    
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
        random.shuffle(file_paths)  # Shuffle to get a random sample
        file_paths = file_paths[:max_files]
    
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
    
    # Create regex baseline first - we'll use this for synthetic annotations
    print("Creating regex baseline...")
    regex_baseline = create_regex_baseline()
    
    # Generate synthetic annotations for training
    print("Generating synthetic annotations for training...")
    annotations = []
    for _, text in samples:
        annotations.append(generate_synthetic_annotations(text, regex_baseline))
    
    # Show annotation counts
    total_annotations = sum(len(a) for a in annotations)
    print(f"Generated {total_annotations} synthetic annotations " +
          f"(avg: {total_annotations/len(annotations):.1f} per file)")
    
    # Create genetic model with seed patterns
    print(f"Creating genetic algorithm model ({generations} generations)...")
    seed_genes = create_seed_patterns()
    print(f"Using {len(seed_genes)} seed patterns for initialization")
    
    # Initialize with seed chromosomes
    seed_chromosome = Chromosome(genes=seed_genes)
    
    # Print out the seed genes for debugging
    print("\nSeed patterns:")
    for i, gene in enumerate(seed_genes):
        print(f"  {i+1}. {gene}")
    
    # Create genetic model with more generations and larger population
    genetic_model = GeneticPIIDetector(
        population_size=100,  # Increased from 50
        generations=generations,
        crossover_prob=0.8,   # Slightly increased from 0.7
        mutation_prob=0.3,    # Increased from 0.2
        gene_mutation_prob=0.4,  # Increased from 0.3
        chromosome_size=len(seed_genes)  # Use all our seed genes
    )
    
    # Train with synthetic annotations
    print("\nTraining genetic algorithm...")
    training_start = time.time()
    
    # Initialize population with seed chromosome
    genetic_model.training_texts = [text for _, text in samples]
    genetic_model.training_annotations = annotations
    
    # Initialize population
    genetic_model.population = genetic_model.toolbox.population()
    
    # Replace first individual with our seed chromosome
    genetic_model.population[0] = [seed_chromosome]
    
    # Evolve population
    genetic_model.population = genetic_model.toolbox.select(
        genetic_model.population, len(genetic_model.population))
    
    for gen in range(genetic_model.generations):
        print(f"\nGeneration {gen+1}/{genetic_model.generations}")
        
        # Select the next generation individuals
        offspring = genetic_model.toolbox.select(
            genetic_model.population, len(genetic_model.population))
        
        # Clone the selected individuals
        offspring = list(map(genetic_model.toolbox.clone, offspring))
        
        # Apply crossover and mutation on the offspring
        for i in range(1, len(offspring), 2):
            if i < len(offspring) - 1:  # Make sure we have a pair
                genetic_model.toolbox.mate(offspring[i], offspring[i + 1])
            
            genetic_model.toolbox.mutate(offspring[i])
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring]
        fitnesses = [genetic_model.toolbox.evaluate(ind) for ind in invalid_ind]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Print best fitness in this generation
        best_fitness = max((ind.fitness.values for ind in offspring), key=lambda x: x[0] + x[1] - x[2])
        print(f"  Best fitness: precision={best_fitness[0]:.3f}, " +
              f"recall={best_fitness[1]:.3f}, complexity={best_fitness[2]:.3f}")
        
        # The population is entirely replaced by the offspring
        genetic_model.population[:] = offspring
    
    # Find best individual
    genetic_model.best_individual = max(genetic_model.population, 
                                       key=lambda ind: ind.fitness.values[0] + ind.fitness.values[1] - ind.fitness.values[2])
    
    training_time = time.time() - training_start
    print(f"\nTraining completed in {training_time:.1f} seconds")
    
    # Print out the best genes after evolution
    best_chromosome = genetic_model.best_individual[0]
    print("\nBest chromosome genes after evolution:")
    for i, gene in enumerate(best_chromosome.genes):
        print(f"  {i+1}. {gene}")
    
    # Process samples in batches
    results = {
        "genetic": [],
        "regex": []
    }
    
    # Split samples into batches
    batches = []
    for i in range(0, len(samples), batch_size):
        batches.append(samples[i:i+batch_size])
    
    # Process each batch
    for i, batch in enumerate(batches):
        batch_results = process_batch(batch, i+1, len(batches), genetic_model, regex_baseline)
        results["genetic"].extend(batch_results["genetic"])
        results["regex"].extend(batch_results["regex"])
    
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
        "entity_types": list(genetic_types),
        "avg_time_seconds": sum(r["time_seconds"] for r in results["genetic"]) / len(results["genetic"])
    }
    
    # Regex model stats
    regex_total = sum(r["count"] for r in results["regex"])
    regex_types = set()
    for r in results["regex"]:
        regex_types.update(r["types"])
    
    summary["regex"] = {
        "total_entities": regex_total,
        "avg_per_file": regex_total / len(samples) if samples else 0,
        "entity_types": list(regex_types),
        "avg_time_seconds": sum(r["time_seconds"] for r in results["regex"]) / len(results["regex"])
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
            "genetic_generations": generations,
            "training_time_seconds": training_time
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
    print(f"  Avg processing time: {summary['genetic']['avg_time_seconds']:.3f} seconds/file")
    
    print(f"\nRegex Baseline:")
    print(f"  Total entities: {summary['regex']['total_entities']}")
    print(f"  Avg per file: {summary['regex']['avg_per_file']:.2f}")
    print(f"  Entity types: {', '.join(summary['regex']['entity_types'])}")
    print(f"  Avg processing time: {summary['regex']['avg_time_seconds']:.3f} seconds/file")
    
    print(f"\nOverlap Analysis:")
    print(f"  Detected by both models: {summary['overlap']['both_models']}")
    print(f"  Only by genetic algorithm: {summary['overlap']['only_genetic']}")
    print(f"  Only by regex baseline: {summary['overlap']['only_regex']}")
    
    print(f"\nResults saved to {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())