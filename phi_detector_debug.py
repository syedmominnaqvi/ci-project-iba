#!/usr/bin/env python
"""
Debug version of PHI detector.
Focuses on diagnosing and fixing batch processing issues.
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

# Import our PHI detection system
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.chromosome import DetectionGene, Chromosome

# Maximum processing time per file
MAX_PROCESSING_TIME = 30  # seconds

class TimeoutException(Exception):
    """Exception raised when a function times out."""
    pass

@contextmanager
def time_limit(seconds):
    """Context manager to limit execution time of a function."""
    def signal_handler(signum, frame):
        raise TimeoutException(f"Function timed out after {seconds} seconds")
    
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

def read_file(file_path, max_size=500000):
    """Read a text file safely with size limit."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read(max_size)
            if len(text) >= max_size:
                print(f"Warning: Truncated file {file_path} to {max_size} characters")
            return text
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def safe_detect(model, text, timeout=MAX_PROCESSING_TIME):
    """
    Safely apply detection with timeout to avoid hanging on problematic files.
    """
    try:
        with time_limit(timeout):
            return model.detect(text)
    except TimeoutException as e:
        print(f"Warning: {e}")
        return []
    except Exception as e:
        print(f"Error during detection: {e}")
        print(traceback.format_exc())
        return []

def process_batch(samples, batch_idx, total_batches, regex_baseline):
    """Process a batch of samples to diagnose issues."""
    print(f"\nProcessing batch {batch_idx}/{total_batches}...")
    
    batch_results = []
    
    # Process each sample in the batch
    for i, (file_path, text) in enumerate(samples):
        file_name = os.path.basename(file_path)
        print(f"  Processing file {i+1}/{len(samples)}: {file_name}")
        
        try:
            # Track timing
            start_time = time.time()
            
            # First check text statistics
            char_count = len(text)
            line_count = text.count('\n') + 1
            print(f"    Text statistics: {char_count} chars, {line_count} lines")
            
            # Regex baseline detection
            print(f"    Running regex baseline...")
            regex_preds = safe_detect(regex_baseline, text)
            regex_time = time.time() - start_time
            
            # Print detection statistics
            print(f"    Found {len(regex_preds)} regex matches in {regex_time:.2f} seconds")
            
            # Record timing info for each regex pattern to identify slow ones
            if regex_preds:
                print(f"    Entity types found: {', '.join(set(p[3] for p in regex_preds))}")
            
            # Test individual patterns to identify problematic ones
            print(f"    Testing individual patterns...")
            for j, gene in enumerate(regex_baseline.genes):
                pattern_start = time.time()
                try:
                    with time_limit(5):  # 5 second timeout per pattern
                        matches = gene.matches(text)
                        pattern_time = time.time() - pattern_start
                        print(f"      Pattern {j+1} ({gene.pii_type}): {len(matches)} matches in {pattern_time:.2f}s")
                except TimeoutException:
                    print(f"      Pattern {j+1} ({gene.pii_type}): TIMED OUT")
                except Exception as e:
                    print(f"      Pattern {j+1} ({gene.pii_type}): ERROR - {str(e)}")
            
            batch_results.append({
                "file": file_name,
                "file_path": file_path,
                "count": len(regex_preds),
                "time_seconds": regex_time,
                "types": list(set(p[3] for p in regex_preds)) if regex_preds else []
            })
            
        except Exception as e:
            print(f"  Error processing file: {e}")
            print(traceback.format_exc())
    
    return batch_results

def main():
    """Run diagnostic PHI detection."""
    try:
        # Parse arguments
        input_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
        max_files = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        
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
        for file_path in file_paths:
            print(f"  Reading {file_path}...")
            text = read_file(file_path)
            if text:
                samples.append((file_path, text))
                print(f"    Read {len(text)} characters")
            else:
                print("    Failed to read file")
        
        if not samples:
            print("No valid text content found!")
            return 1
        
        print(f"Successfully read {len(samples)} files")
        
        # Create regex baseline
        print("Creating regex baseline...")
        regex_baseline = create_regex_baseline()
        
        # Process samples in batches
        results = []
        
        # Split samples into batches
        batches = []
        for i in range(0, len(samples), batch_size):
            batches.append(samples[i:i+batch_size])
        
        # Process each batch
        for i, batch in enumerate(batches):
            try:
                print(f"\nStarting batch {i+1}/{len(batches)} with {len(batch)} files")
                batch_results = process_batch(batch, i+1, len(batches), regex_baseline)
                results.extend(batch_results)
                print(f"Completed batch {i+1}/{len(batches)}")
            except Exception as e:
                print(f"Error processing batch {i+1}: {e}")
                print(traceback.format_exc())
        
        # Print summary
        print("\nDIAGNOSTIC SUMMARY:")
        print(f"  Files processed: {len(results)}/{len(samples)}")
        print(f"  Total entities found: {sum(r['count'] for r in results)}")
        print(f"  Avg entities per file: {sum(r['count'] for r in results)/len(results) if results else 0:.2f}")
        print(f"  Avg processing time: {sum(r['time_seconds'] for r in results)/len(results) if results else 0:.3f} seconds/file")
        
        # Save diagnostic results
        output_file = "phi_debug_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "files_processed": len(results),
                },
                "results": results
            }, f, indent=2, default=str)
        
        print(f"\nDiagnostic results saved to {output_file}")
        return 0
    
    except Exception as e:
        print(f"Fatal error: {e}")
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())