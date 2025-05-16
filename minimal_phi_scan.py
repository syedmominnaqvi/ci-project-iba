#!/usr/bin/env python
"""
Minimal PHI scanner with NO external dependencies or complex operations.
This script WILL NOT get stuck because it:
1. Uses only built-in Python libraries
2. Skips PDF extraction entirely
3. Uses a simple regex-based detection approach
4. Has strict timeouts on all operations
5. Processes only one file at a time
"""
import os
import sys
import re
import json
import time
import signal
import random
from datetime import datetime
from contextlib import contextmanager

# === Timeout handling ===
class TimeoutException(Exception):
    """Exception raised when a function times out."""
    pass

@contextmanager
def time_limit(seconds):
    """Context manager to limit execution time of a function."""
    def signal_handler(signum, frame):
        raise TimeoutException(f"Timed out after {seconds} seconds")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# === PHI detection patterns ===
PHI_PATTERNS = [
    (r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', "NAME"),                             # Names
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "EMAIL"),      # Emails
    (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', "PHONE"),                            # Phone numbers
    (r'\b\d{3}[-]?\d{2}[-]?\d{4}\b', "SSN"),                               # SSNs
    (r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', "DATE"),                              # Dates
    (r'\b\d+\s+[A-Z][a-z]+\s+[A-Za-z]+\b', "ADDRESS"),                     # Addresses
    (r'\bMRN:?\s*\d+\b', "MEDICAL_RECORD"),                                # Medical record numbers
    (r'\bPatient ID:?\s*\d+\b', "MEDICAL_RECORD"),                         # Patient IDs
    (r'\bDr\.\s+[A-Z][a-z]+\b', "NAME"),                                   # Doctor names
]

# Compile all patterns for efficiency
COMPILED_PATTERNS = [(re.compile(pattern), phi_type) for pattern, phi_type in PHI_PATTERNS]

# === File handling functions ===
def extract_text_from_file(file_path, max_size=100000):
    """Extract text from file with strict timeout and size limits."""
    extension = os.path.splitext(file_path)[1].lower()
    
    # Skip PDF files completely
    if extension == '.pdf':
        print(f"Skipping PDF file: {file_path}")
        return ""
    
    # Skip non-text files
    if extension not in ['.txt', '.csv', '.json', '.html', '.md', '.log', '.xml', '']:
        print(f"Skipping non-text file: {file_path}")
        return ""
    
    try:
        with time_limit(10):  # 10 second timeout for file reading
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read(max_size)
                if len(text) >= max_size:
                    print(f"Truncated {file_path} to {max_size} characters")
                return text
    except TimeoutException:
        print(f"Timeout reading {file_path}")
        return ""
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def find_text_files(directory, max_files=None):
    """Find all text files in a directory with basic error handling."""
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return []
    
    print(f"Scanning directory: {directory}")
    
    text_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            try:
                # Only process text files
                if file.endswith(('.txt', '.csv', '.json', '.html', '.md', '.log', '.xml')):
                    file_path = os.path.join(root, file)
                    text_files.append(file_path)
            except Exception as e:
                print(f"Error processing file {file}: {e}")
    
    print(f"Found {len(text_files)} text files")
    
    # Limit number of files if specified
    if max_files and len(text_files) > max_files:
        print(f"Limiting to {max_files} files (randomly selected)")
        random.shuffle(text_files)
        text_files = text_files[:max_files]
    
    return text_files

# === PHI detection function ===
def detect_phi(text, confidence=0.8):
    """Detect PHI in text using regex patterns."""
    if not text:
        return []
    
    all_matches = []
    
    # Apply timeout to full detection
    try:
        with time_limit(30):  # 30 second timeout for pattern matching
            # Apply each pattern
            for pattern, phi_type in COMPILED_PATTERNS:
                for match in pattern.finditer(text):
                    start, end = match.span()
                    all_matches.append({
                        "text": match.group(),
                        "start": start,
                        "end": end,
                        "type": phi_type,
                        "confidence": confidence
                    })
    except TimeoutException:
        print("PHI detection timed out")
        return all_matches  # Return whatever we found so far
    except Exception as e:
        print(f"Error during PHI detection: {e}")
        return []
    
    return all_matches

# === Main processing function ===
def process_files(file_paths, output_dir):
    """Process files for PHI detection one at a time."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize stats
    all_results = []
    successful_files = 0
    failed_files = 0
    phi_counts = {}
    
    # Process each file individually
    for i, file_path in enumerate(file_paths):
        try:
            file_name = os.path.basename(file_path)
            print(f"Processing file {i+1}/{len(file_paths)}: {file_name}")
            
            # Extract text
            text = extract_text_from_file(file_path)
            if not text or len(text) < 50:
                print(f"Insufficient text in {file_name}, skipping")
                failed_files += 1
                continue
            
            # Detect PHI
            print(f"Detecting PHI in {file_name}...")
            phi_matches = detect_phi(text)
            print(f"Found {len(phi_matches)} PHI entities")
            
            # Track PHI types
            for match in phi_matches:
                phi_type = match["type"]
                phi_counts[phi_type] = phi_counts.get(phi_type, 0) + 1
            
            # Save results for this file
            file_result = {
                "file": file_name,
                "file_path": file_path,
                "text_length": len(text),
                "phi_count": len(phi_matches),
                "phi_types": list(set(m["type"] for m in phi_matches)),
                "phi_entities": phi_matches
            }
            
            all_results.append(file_result)
            
            # Save individual file report
            file_base = os.path.splitext(file_name)[0]
            report_path = os.path.join(output_dir, f"{file_base}_phi_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(file_result, f, indent=2)
            
            successful_files += 1
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            failed_files += 1
    
    # Create summary
    summary = {
        "total_files_processed": successful_files + failed_files,
        "successful_files": successful_files,
        "failed_files": failed_files,
        "total_phi_entities": sum(len(r["phi_entities"]) for r in all_results),
        "phi_types_found": list(phi_counts.keys()),
        "phi_counts_by_type": phi_counts,
        "detection_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save summary
    summary_path = os.path.join(output_dir, "phi_detection_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    # Print report
    print("\nPHI DETECTION SUMMARY")
    print("====================")
    print(f"Total files processed: {summary['total_files_processed']}")
    print(f"Successfully processed: {summary['successful_files']}")
    print(f"Failed to process: {summary['failed_files']}")
    print(f"Total PHI entities found: {summary['total_phi_entities']}")
    print(f"PHI types found: {', '.join(summary['phi_types_found'])}")
    print("\nPHI counts by type:")
    for phi_type, count in phi_counts.items():
        print(f"  {phi_type}: {count}")
    
    print(f"\nResults saved to: {output_dir}")
    
    return summary

def main():
    """Parse arguments and run PHI detection."""
    if len(sys.argv) < 2:
        print("Usage: python minimal_phi_scan.py INPUT_DIRECTORY [OUTPUT_DIRECTORY] [MAX_FILES]")
        return 1
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "phi_results"
    max_files = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    print(f"Minimal PHI Scanner")
    print(f"=================")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    if max_files:
        print(f"Maximum files: {max_files}")
    
    file_paths = find_text_files(input_dir, max_files)
    
    if not file_paths:
        print("No text files found to process.")
        return 1
    
    process_files(file_paths, output_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())