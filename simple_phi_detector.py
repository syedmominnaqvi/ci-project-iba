#!/usr/bin/env python
"""
Simple PHI/PII detector for text files only.
This script handles only text files and uses the genetic algorithm
for PHI detection with minimal dependencies.
"""
import os
import sys
import argparse
import json
import re
import random
from datetime import datetime

# Import our PHI detection system
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from src.genetic_algorithm import GeneticPIIDetector
    from src.chromosome import DetectionGene, Chromosome
except ImportError:
    print("Error: Could not import required modules from src/")
    print("Make sure you are running this script from the project root directory.")
    sys.exit(1)


def find_text_files(input_dir, recursive=True):
    """
    Find all text files in a directory.
    
    Args:
        input_dir: Directory to search
        recursive: Whether to search subdirectories
        
    Returns:
        List of file paths
    """
    text_files = []
    
    # Extensions that are likely to contain readable text
    text_extensions = ['.txt', '.csv', '.json', '.md', '.log', '.xml', '.html']
    
    # Walk through directory
    for root, dirs, files in os.walk(input_dir):
        if not recursive and root != input_dir:
            continue
            
        for file in files:
            # Check if it's a text file by extension
            ext = os.path.splitext(file)[1].lower()
            if ext in text_extensions:
                file_path = os.path.join(root, file)
                text_files.append(file_path)
    
    return text_files


def is_text_file(file_path, min_ratio=0.7, sample_size=1000):
    """
    Check if a file is likely to be a text file by examining its content.
    
    Args:
        file_path: Path to the file
        min_ratio: Minimum ratio of printable ASCII characters
        sample_size: Number of bytes to sample
        
    Returns:
        True if the file is likely a text file, False otherwise
    """
    try:
        # Read a sample of the file
        with open(file_path, 'rb') as f:
            sample = f.read(sample_size)
        
        # Count printable ASCII characters
        printable_count = sum(32 <= byte <= 126 for byte in sample)
        
        # Calculate ratio of printable characters
        ratio = printable_count / len(sample) if sample else 0
        
        return ratio >= min_ratio
    except Exception:
        return False


def find_readable_files(input_dir, recursive=True, include_unknown=True):
    """
    Find all potentially readable files in a directory.
    
    Args:
        input_dir: Directory to search
        recursive: Whether to search subdirectories
        include_unknown: Whether to include files with unknown extensions
        
    Returns:
        List of file paths
    """
    known_text_files = find_text_files(input_dir, recursive)
    
    # If we don't want to include unknown files, just return known text files
    if not include_unknown:
        return known_text_files
    
    # Otherwise, check other files
    other_files = []
    
    for root, dirs, files in os.walk(input_dir):
        if not recursive and root != input_dir:
            continue
            
        for file in files:
            # Skip files we already found
            file_path = os.path.join(root, file)
            if file_path in known_text_files:
                continue
            
            # Check if it's a text file by content
            if is_text_file(file_path):
                other_files.append(file_path)
    
    return known_text_files + other_files


def read_text_file(file_path):
    """
    Read a text file safely.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File content as string, or empty string on error
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def get_synthetic_annotations(text):
    """
    Generate synthetic annotations for training using regex patterns.
    
    Args:
        text: Text to annotate
        
    Returns:
        List of (start, end, type) tuples
    """
    annotations = []
    
    patterns = [
        (r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', "NAME"),  # Names
        (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', "PHONE"),  # Phone numbers
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "EMAIL"),  # Emails
        (r'\b\d{3}[-]?\d{2}[-]?\d{4}\b', "SSN"),  # SSNs
        (r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', "DATE"),  # Dates
        (r'\b(?:MR|MRN)[-: ]*\d+\b', "MEDICAL_RECORD")  # Medical record numbers
    ]
    
    for pattern, phi_type in patterns:
        for match in re.finditer(pattern, text):
            annotations.append((match.start(), match.end(), phi_type))
    
    return annotations


def create_simple_baseline_model():
    """
    Create a simple baseline model using regex patterns.
    
    Returns:
        A Chromosome object that can detect PHI
    """
    genes = [
        DetectionGene(pattern=r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', pii_type="NAME"),
        DetectionGene(pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', pii_type="EMAIL"),
        DetectionGene(pattern=r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', pii_type="PHONE"),
        DetectionGene(pattern=r'\b\d{3}[-]?\d{2}[-]?\d{4}\b', pii_type="SSN"),
        DetectionGene(pattern=r'\b\d+\s+[A-Z][a-z]+\s+[A-Za-z]+\b', pii_type="ADDRESS"),
        DetectionGene(pattern=r'\b(?:MR|MRN)[-: ]*\d+\b', pii_type="MEDICAL_RECORD"),
        DetectionGene(pattern=r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', pii_type="DATE"),
    ]
    
    baseline = Chromosome(genes=genes)
    return baseline


def process_files(file_paths, output_dir, generations=20, batch_size=10):
    """
    Process files for PHI detection.
    
    Args:
        file_paths: List of file paths to process
        output_dir: Directory to save results
        generations: Number of generations for genetic algorithm
        batch_size: Number of files to process in each batch
        
    Returns:
        Dictionary with results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize models
    print(f"Initializing PHI detection models...")
    
    # Genetic algorithm model
    genetic_model = GeneticPIIDetector(
        population_size=30,
        generations=generations,
        crossover_prob=0.7,
        mutation_prob=0.2
    )
    
    # Simple baseline for comparison
    baseline_model = create_simple_baseline_model()
    
    # Process files in batches
    all_results = []
    
    for batch_idx in range(0, len(file_paths), batch_size):
        batch_files = file_paths[batch_idx:batch_idx + batch_size]
        
        print(f"\nProcessing batch {batch_idx//batch_size + 1}/{(len(file_paths) + batch_size - 1)//batch_size}")
        
        # Extract text from each file
        batch_texts = []
        extracted_info = []
        
        for file_path in batch_files:
            print(f"Reading {file_path}")
            text = read_text_file(file_path)
            
            # Skip files with no content
            if not text or len(text) < 50:
                print(f"Skipping {file_path}: insufficient text content")
                continue
            
            batch_texts.append(text)
            extracted_info.append({
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "file_size": os.path.getsize(file_path),
                "text_length": len(text)
            })
        
        if not batch_texts:
            print("No valid text extracted in this batch, skipping")
            continue
        
        # Generate synthetic annotations for training
        synthetic_annotations = []
        for text in batch_texts:
            annotations = get_synthetic_annotations(text)
            synthetic_annotations.append(annotations)
        
        # Train the genetic model
        print(f"Training genetic algorithm on {len(batch_texts)} documents ({sum(len(t) for t in batch_texts)/1000:.1f}K characters)...")
        genetic_model.train(batch_texts, synthetic_annotations)
        
        # Process each file
        batch_results = []
        
        for i, text in enumerate(batch_texts):
            file_info = extracted_info[i]
            
            # Genetic algorithm detection
            genetic_preds = genetic_model.predict(text)
            
            # Baseline detection
            baseline_preds = baseline_model.detect(text)
            
            # Combine results
            result = {
                "file_path": file_info["file_path"],
                "file_name": file_info["file_name"],
                "file_size": file_info["file_size"],
                "text_length": file_info["text_length"],
                "genetic_detections": genetic_preds,
                "genetic_count": len(genetic_preds),
                "baseline_detections": baseline_preds,
                "baseline_count": len(baseline_preds),
                "phi_types_genetic": list(set(p[3] for p in genetic_preds)),
                "phi_types_baseline": list(set(p[3] for p in baseline_preds))
            }
            
            # Print detected PHI
            print(f"\nFile: {file_info['file_name']}")
            print(f"PHI detected (genetic): {len(genetic_preds)} instances")
            print(f"PHI detected (baseline): {len(baseline_preds)} instances")
            
            if genetic_preds:
                print("Sample PHI instances:")
                for match, start, end, phi_type, confidence in genetic_preds[:5]:
                    print(f"  - {phi_type}: '{match}' (confidence: {confidence:.2f})")
                if len(genetic_preds) > 5:
                    print(f"  - ... and {len(genetic_preds) - 5} more")
            
            # Save results for this file
            batch_results.append(result)
            
            # Generate a detailed report for this file
            file_report = {
                "file_info": file_info,
                "genetic_detections": [{
                    "text": p[0],
                    "start": p[1],
                    "end": p[2],
                    "type": p[3],
                    "confidence": p[4]
                } for p in genetic_preds],
                "baseline_detections": [{
                    "text": p[0],
                    "start": p[1],
                    "end": p[2],
                    "type": p[3],
                    "confidence": p[4]
                } for p in baseline_preds],
                "detection_summary": {
                    "genetic": {
                        "total": len(genetic_preds),
                        "by_type": {t: len([p for p in genetic_preds if p[3] == t]) for t in set(p[3] for p in genetic_preds)}
                    },
                    "baseline": {
                        "total": len(baseline_preds),
                        "by_type": {t: len([p for p in baseline_preds if p[3] == t]) for t in set(p[3] for p in baseline_preds)}
                    }
                },
                "text_sample": text[:1000] + ("..." if len(text) > 1000 else "")
            }
            
            # Save detailed report to file
            file_base = os.path.splitext(file_info["file_name"])[0]
            report_path = os.path.join(output_dir, f"{file_base}_phi_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(file_report, f, indent=2)
        
        all_results.extend(batch_results)
    
    # Generate summary report
    if all_results:
        summary = {
            "total_files": len(all_results),
            "total_phi_detected": {
                "genetic": sum(r["genetic_count"] for r in all_results),
                "baseline": sum(r["baseline_count"] for r in all_results)
            },
            "files_with_phi": {
                "genetic": sum(1 for r in all_results if r["genetic_count"] > 0),
                "baseline": sum(1 for r in all_results if r["baseline_count"] > 0)
            },
            "phi_types_found": {
                "genetic": list(set(typ for r in all_results for typ in r["phi_types_genetic"])),
                "baseline": list(set(typ for r in all_results for typ in r["phi_types_baseline"]))
            },
            "top_files_by_phi_count": sorted(
                [{"file_name": r["file_name"], "phi_count": r["genetic_count"]} for r in all_results],
                key=lambda x: x["phi_count"],
                reverse=True
            )[:10],
            "detection_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_settings": {
                "genetic_generations": generations
            }
        }
        
        # Save summary
        summary_path = os.path.join(output_dir, "phi_detection_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nPHI Detection Summary:")
        print(f"Total files processed: {summary['total_files']}")
        print(f"Files with PHI detected (genetic): {summary['files_with_phi']['genetic']}")
        print(f"Files with PHI detected (baseline): {summary['files_with_phi']['baseline']}")
        print(f"Total PHI instances found (genetic): {summary['total_phi_detected']['genetic']}")
        print(f"Total PHI instances found (baseline): {summary['total_phi_detected']['baseline']}")
        print(f"\nPHI types found (genetic): {', '.join(summary['phi_types_found']['genetic'])}")
        print(f"\nResults saved to: {output_dir}")
        
        return summary
    else:
        print("No valid results found.")
        return None


def main():
    """Parse arguments and run PHI detection."""
    parser = argparse.ArgumentParser(description="Simple PHI detector for text files")
    
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory to scan")
    parser.add_argument("--output", type=str, default="phi_results",
                        help="Output directory for results")
    parser.add_argument("--recursive", action="store_true", default=True,
                        help="Search subdirectories recursively")
    parser.add_argument("--generations", type=int, default=10,
                        help="Number of generations for genetic algorithm")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Number of files to process in each batch")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Input directory not found: {args.input}")
        return 1
    
    print(f"Scanning directory for text files: {args.input}")
    file_paths = find_readable_files(args.input, args.recursive)
    
    if not file_paths:
        print("No text files found.")
        return 1
    
    print(f"Found {len(file_paths)} readable text files.")
    
    # Process files
    process_files(
        file_paths,
        args.output,
        generations=args.generations,
        batch_size=args.batch_size
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())