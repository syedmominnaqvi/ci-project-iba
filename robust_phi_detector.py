#!/usr/bin/env python
"""
Robust PHI detector with improved error handling and batch processing.
This version adds timeouts, better debugging, and enhanced genetic algorithm.
"""
import os
import sys
import json
import glob
import random
import re
import time
import signal
import traceback
from datetime import datetime
from contextlib import contextmanager
from tqdm import tqdm

# Import our PHI detection system
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.genetic_algorithm import GeneticPIIDetector
from src.chromosome import DetectionGene, Chromosome

# Maximum text size to process (to avoid memory issues)
MAX_TEXT_SIZE = 500_000  # characters
MAX_PROCESSING_TIME = 60  # seconds per file

class TimeoutException(Exception):
    """Exception raised when a function times out."""
    pass

@contextmanager
def time_limit(seconds):
    """Context manager to limit execution time of a function."""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

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
        DetectionGene(pattern=r'\bDr\.\s+[A-Z][a-z]+\b', pii_type="NAME", confidence=0.8),
        
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
        
        # Addresses
        DetectionGene(pattern=r'\b\d+\s+[A-Z][a-z]+\s+(St|Ave|Rd|Blvd|Drive|Lane|Court|Way)\b', pii_type="ADDRESS", confidence=0.8),
        
        # Medical record numbers
        DetectionGene(pattern=r'\bMRN:?\s*\d+\b', pii_type="MEDICAL_RECORD", confidence=0.9),
        DetectionGene(pattern=r'\bPatient ID:?\s*\d+\b', pii_type="MEDICAL_RECORD", confidence=0.9),
        DetectionGene(pattern=r'\bRecord #\s*\d+\b', pii_type="MEDICAL_RECORD", confidence=0.8),
    ]
    return genes


def find_text_files(directory, max_files=None):
    """Find all text files in a directory."""
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return []
    
    text_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                text_files.append(os.path.join(root, file))
    
    # Limit number of files if needed
    if max_files and len(text_files) > max_files:
        print(f"Limiting to {max_files} random files")
        random.shuffle(text_files)  # Shuffle to get a random sample
        text_files = text_files[:max_files]
    
    return text_files


def read_file(file_path, max_size=MAX_TEXT_SIZE):
    """Read a text file safely."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read(max_size)
            if len(text) >= max_size:
                print(f"Warning: Truncated file {file_path} to {max_size} characters")
            return text
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def generate_synthetic_annotations(text, regex_baseline):
    """
    Generate synthetic annotations based on regex patterns.
    This helps provide training examples for the genetic algorithm.
    """
    try:
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
    except Exception as e:
        print(f"Error generating annotations: {e}")
        return []


def safe_detect(model, text, timeout=MAX_PROCESSING_TIME):
    """
    Safely apply detection with timeout to avoid hanging on problematic files.
    """
    try:
        with time_limit(timeout):
            if isinstance(model, GeneticPIIDetector):
                return model.predict(text)
            else:  # Chromosome
                return model.detect(text)
    except TimeoutException:
        print(f"Detection timed out after {timeout} seconds")
        return []
    except Exception as e:
        print(f"Error during detection: {e}")
        print(traceback.format_exc())
        return []


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
        genetic_preds = safe_detect(genetic_model, text)
        genetic_time = time.time() - start_time
        
        # Regex baseline detection
        print(f"    Running regex baseline...")
        start_time = time.time()
        regex_preds = safe_detect(regex_baseline, text)
        regex_time = time.time() - start_time
        
        # Print detection counts
        print(f"    Found {len(genetic_preds)} genetic matches, {len(regex_preds)} regex matches")
        
        # Record results
        batch_results["genetic"].append({
            "file": file_name,
            "file_path": file_path,
            "count": len(genetic_preds),
            "types": list(set(p[3] for p in genetic_preds)) if genetic_preds else [],
            "time_seconds": genetic_time,
            "entities": [{"text": p[0], "type": p[3], "confidence": float(p[4])} for p in genetic_preds]
        })
        
        batch_results["regex"].append({
            "file": file_name,
            "file_path": file_path,
            "count": len(regex_preds),
            "types": list(set(p[3] for p in regex_preds)) if regex_preds else [],
            "time_seconds": regex_time,
            "entities": [{"text": p[0], "type": p[3], "confidence": float(p[4])} for p in regex_preds]
        })
    
    return batch_results


def main():
    """Run robust PHI detection comparison."""
    try:
        # Parse arguments
        input_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
        output_file = sys.argv[2] if len(sys.argv) > 2 else "robust_phi_results.json"
        generations = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        max_files = int(sys.argv[4]) if len(sys.argv) > 4 else 20
        batch_size = int(sys.argv[5]) if len(sys.argv) > 5 else 5
        
        # Find text files
        print(f"Finding text files in {input_dir}...")
        file_paths = find_text_files(input_dir, max_files)
        
        if not file_paths:
            print("No text files found!")
            return 1
        
        print(f"Found {len(file_paths)} text files")
        
        # Read files
        print("Reading files...")
        samples = []
        for file_path in tqdm(file_paths, desc="Extracting text"):
            text = read_file(file_path)
            if text:
                samples.append((file_path, text))
        
        if not samples:
            print("No valid text content found!")
            return 1
        
        print(f"Successfully read {len(samples)} files")
        print(f"Total text size: {sum(len(text) for _, text in samples) / 1000:.1f}K characters")
        
        # Create regex baseline first - we'll use this for synthetic annotations
        print("Creating regex baseline...")
        regex_baseline = create_regex_baseline()
        
        # Generate synthetic annotations for training
        print("Generating synthetic annotations for training...")
        annotations = []
        for _, text in tqdm(samples, desc="Generating training annotations"):
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
        
        # Create genetic model with better parameters
        genetic_model = GeneticPIIDetector(
            population_size=50,
            generations=generations,
            crossover_prob=0.7,
            mutation_prob=0.3,
            gene_mutation_prob=0.3,
            chromosome_size=len(seed_genes)
        )
        
        # Train with synthetic annotations
        print(f"\nTraining genetic algorithm on {len(samples)} documents " +
              f"({sum(len(text) for _, text in samples) / 1000:.1f}K characters)...")
        
        try:
            genetics_texts = [text for _, text in samples]
            genetic_model.train(genetics_texts, annotations)
        except Exception as e:
            print(f"Error during training: {e}")
            print(traceback.format_exc())
            
            # If training fails, at least use our seed chromosome
            genetic_model.best_individual = [seed_chromosome]
            print("Using seed chromosome as fallback due to training error")
        
        # Process samples in batches
        results = {
            "genetic": [],
            "regex": []
        }
        
        # Split samples into batches
        batches = []
        for i in range(0, len(samples), batch_size):
            batches.append(samples[i:i+batch_size])
        
        # Process each batch with error handling
        for i, batch in enumerate(batches):
            try:
                batch_results = process_batch(batch, i+1, len(batches), genetic_model, regex_baseline)
                results["genetic"].extend(batch_results["genetic"])
                results["regex"].extend(batch_results["regex"])
            except Exception as e:
                print(f"Error processing batch {i+1}: {e}")
                print(traceback.format_exc())
                # Continue with next batch
        
        # Calculate summary
        summary = {}
        
        # Genetic model stats
        genetic_total = sum(r["count"] for r in results["genetic"])
        genetic_types = set()
        for r in results["genetic"]:
            if "types" in r:
                genetic_types.update(r["types"])
        
        summary["genetic"] = {
            "total_entities": genetic_total,
            "avg_per_file": genetic_total / len(samples) if samples else 0,
            "entity_types": list(genetic_types),
            "avg_time_seconds": sum(r.get("time_seconds", 0) for r in results["genetic"]) / len(results["genetic"]) if results["genetic"] else 0
        }
        
        # Regex model stats
        regex_total = sum(r["count"] for r in results["regex"])
        regex_types = set()
        for r in results["regex"]:
            if "types" in r:
                regex_types.update(r["types"])
        
        summary["regex"] = {
            "total_entities": regex_total,
            "avg_per_file": regex_total / len(samples) if samples else 0,
            "entity_types": list(regex_types),
            "avg_time_seconds": sum(r.get("time_seconds", 0) for r in results["regex"]) / len(results["regex"]) if results["regex"] else 0
        }
        
        # Calculate overlap
        overlap_count = 0
        only_genetic = 0
        only_regex = 0
        
        for i in range(len(samples)):
            if i < len(results["genetic"]) and i < len(results["regex"]):
                genetic_entities = {e["text"] for e in results["genetic"][i].get("entities", [])}
                regex_entities = {e["text"] for e in results["regex"][i].get("entities", [])}
                
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
    
    except Exception as e:
        print(f"Fatal error: {e}")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())