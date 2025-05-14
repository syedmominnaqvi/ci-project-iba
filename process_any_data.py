#!/usr/bin/env python
"""
Process any data directory for PHI/PII detection.
This script:
1. Recursively scans any directory structure
2. Extracts text from various file formats (txt, pdf, doc, etc.)
3. Runs PHI detection on all extracted content
4. Reports findings in a consolidated report
"""
import os
import sys
import argparse
import json
import glob
import tempfile
import subprocess
import random
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd

# Check for optional dependencies
TQDM_AVAILABLE = False
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    # Define a simple replacement if tqdm is not available
    def tqdm(iterable, **kwargs):
        if "desc" in kwargs:
            print(f"{kwargs['desc']}...")
        return iterable

# Import our PHI detection system
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.genetic_algorithm import GeneticPIIDetector
try:
    from src.baseline import PresidioBaseline
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False


def extract_text_from_file(file_path):
    """
    Extract text from various file formats.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Extracted text or empty string if extraction fails
    """
    file_path = str(file_path)
    extension = os.path.splitext(file_path)[1].lower()
    
    # Plain text files
    if extension in ['.txt', '.csv', '.json', '.xml', '.html', '.md', '.log']:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""
    
    # PDF files
    elif extension == '.pdf':
        try:
            # Try using PyPDF2 which doesn't require external tools
            try:
                import PyPDF2
                with open(file_path, 'rb') as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    text = ""
                    for page_num in range(len(reader.pages)):
                        text += reader.pages[page_num].extract_text() + "\n"
                    if text.strip():
                        return text
            except (ImportError, Exception) as e:
                print(f"PyPDF2 extraction failed: {e}")

            # Try using pdfplumber which has better extraction
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                    if text.strip():
                        return text
            except (ImportError, Exception) as e:
                print(f"pdfplumber extraction failed: {e}")

            # Try command-line tools as last resort
            try:
                # Try pdftotext if available
                result = subprocess.run(
                    ['pdftotext', file_path, '-'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    timeout=30
                )
                if result.returncode == 0 and result.stdout:
                    return result.stdout

                # Fallback to pdf2txt.py if available
                result = subprocess.run(
                    ['pdf2txt.py', file_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    timeout=30
                )
                if result.returncode == 0 and result.stdout:
                    return result.stdout
            except Exception:
                pass

            # Return placeholder if all methods failed
            print(f"All PDF extraction methods failed for {file_path}")
            return f"[PDF FILE: {os.path.basename(file_path)}] - PDF extraction failed. Try installing PyPDF2 or pdfplumber."

        except Exception as e:
            print(f"Error extracting text from PDF {file_path}: {e}")
            return f"[PDF FILE: {os.path.basename(file_path)}] - Error during extraction"
    
    # Microsoft Office documents
    elif extension in ['.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']:
        try:
            # Try textract if available
            import textract
            text = textract.process(file_path).decode('utf-8', errors='ignore')
            return text
        except ImportError:
            try:
                # Try Apache Tika if available
                import tika
                from tika import parser
                tika.initVM()
                parsed = parser.from_file(file_path)
                if parsed and 'content' in parsed and parsed['content']:
                    return parsed['content']
            except ImportError:
                return f"[OFFICE FILE: {os.path.basename(file_path)}] - Install textract or tika for content extraction"
        except Exception as e:
            print(f"Error extracting text from Office file {file_path}: {e}")
            return f"[OFFICE FILE: {os.path.basename(file_path)}] - Error during extraction"
    
    # Email files
    elif extension in ['.eml', '.msg']:
        try:
            # Try email extraction if available
            import email
            with open(file_path, 'rb') as f:
                msg = email.message_from_binary_file(f)
                text = ""
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        text += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                return text
        except ImportError:
            return f"[EMAIL FILE: {os.path.basename(file_path)}] - Install email library for content extraction"
        except Exception as e:
            print(f"Error extracting text from email {file_path}: {e}")
            return f"[EMAIL FILE: {os.path.basename(file_path)}] - Error during extraction"
    
    # Default case for unsupported formats
    else:
        # Try to read as text anyway for unknown formats
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # Check if it looks like text
                if '\0' not in content and len(content.strip()) > 0:
                    return content
        except:
            pass
        
        return f"[UNSUPPORTED FILE: {os.path.basename(file_path)}] - Format not supported for extraction"


def scan_directory(input_dir, file_types=None, max_files=None, min_size=10, max_size=10*1024*1024):
    """
    Recursively scan directory for files to process.
    
    Args:
        input_dir: Directory to scan
        file_types: List of file extensions to include (None = all)
        max_files: Maximum number of files to process (None = all)
        min_size: Minimum file size in bytes
        max_size: Maximum file size in bytes
        
    Returns:
        List of file paths
    """
    if not os.path.exists(input_dir):
        print(f"Directory not found: {input_dir}")
        return []
    
    print(f"Scanning directory: {input_dir}")
    
    all_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Check file size
            try:
                file_size = os.path.getsize(file_path)
                if file_size < min_size or file_size > max_size:
                    continue
            except:
                continue
            
            # Check file extension if specified
            if file_types:
                ext = os.path.splitext(file)[1].lower()
                if ext not in file_types:
                    continue
            
            all_files.append(file_path)
    
    print(f"Found {len(all_files)} files matching criteria")
    
    # Limit number of files if specified
    if max_files and len(all_files) > max_files:
        print(f"Limiting to {max_files} files (randomly selected)")
        all_files = random.sample(all_files, max_files)
    
    return all_files


def process_files(file_paths, output_dir, generations=20, use_presidio=True, batch_size=10):
    """
    Process files for PHI detection.
    
    Args:
        file_paths: List of file paths to process
        output_dir: Directory to save results
        generations: Number of generations for genetic algorithm
        use_presidio: Whether to use Presidio baseline for comparison
        batch_size: Number of files to process in each batch
        
    Returns:
        DataFrame with results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize models
    print("Initializing PHI detection models...")
    
    # Genetic algorithm model
    genetic_model = GeneticPIIDetector(
        population_size=50,
        generations=generations,
        crossover_prob=0.7,
        mutation_prob=0.2
    )
    
    # Presidio baseline if available
    presidio_model = None
    if use_presidio and PRESIDIO_AVAILABLE:
        try:
            presidio_model = PresidioBaseline()
            print("Presidio baseline initialized successfully")
        except Exception as e:
            print(f"Error initializing Presidio: {e}")
            presidio_model = None
    
    # Process files in batches
    all_results = []
    
    for batch_idx in range(0, len(file_paths), batch_size):
        batch_files = file_paths[batch_idx:batch_idx + batch_size]
        
        print(f"\nProcessing batch {batch_idx//batch_size + 1}/{(len(file_paths) + batch_size - 1)//batch_size}")
        
        # Extract text from each file
        batch_texts = []
        extracted_info = []
        
        for file_path in tqdm(batch_files, desc="Extracting text"):
            text = extract_text_from_file(file_path)
            
            # Skip files with no content
            if not text or len(text) < 50:
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
        
        # Train genetic model on this batch
        print(f"Training genetic algorithm on {len(batch_texts)} documents ({sum(len(t) for t in batch_texts)/1000:.1f}K characters)...")
        
        # Generate synthetic annotations for training (since we don't have ground truth)
        # This is just to bootstrap the genetic algorithm's learning process
        synthetic_annotations = []
        
        # Use Presidio to generate initial annotations if available
        if presidio_model:
            for text in tqdm(batch_texts, desc="Generating training annotations"):
                presidio_results = presidio_model.detect(text)
                annotations = [(start, end, pii_type) for _, start, end, pii_type, _ in presidio_results]
                synthetic_annotations.append(annotations)
        else:
            # Simple pattern-based annotations as fallback
            patterns = [
                (r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', "NAME"),  # Names
                (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', "PHONE"),  # Phone numbers
                (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "EMAIL"),  # Emails
                (r'\b\d{3}[-]?\d{2}[-]?\d{4}\b', "SSN"),  # SSNs
            ]
            
            import re
            for text in tqdm(batch_texts, desc="Generating training annotations"):
                annotations = []
                for pattern, pii_type in patterns:
                    for match in re.finditer(pattern, text):
                        annotations.append((match.start(), match.end(), pii_type))
                synthetic_annotations.append(annotations)
        
        # Train the genetic model
        genetic_model.train(batch_texts, synthetic_annotations)
        
        # Run detection with both models
        batch_results = []
        
        for i, text in enumerate(tqdm(batch_texts, desc="Detecting PHI")):
            file_info = extracted_info[i]
            
            # Genetic algorithm detection
            genetic_preds = genetic_model.predict(text)
            
            # Presidio detection if available
            presidio_preds = []
            if presidio_model:
                presidio_preds = presidio_model.detect(text)
            
            # Combine results
            result = {
                "file_path": file_info["file_path"],
                "file_name": file_info["file_name"],
                "file_size": file_info["file_size"],
                "text_length": file_info["text_length"],
                "genetic_detections": genetic_preds,
                "genetic_count": len(genetic_preds),
                "presidio_detections": presidio_preds,
                "presidio_count": len(presidio_preds),
                "phi_types_genetic": list(set(p[3] for p in genetic_preds)),
                "phi_types_presidio": list(set(p[3] for p in presidio_preds)) if presidio_preds else []
            }
            
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
                "presidio_detections": [{
                    "text": p[0],
                    "start": p[1],
                    "end": p[2],
                    "type": p[3],
                    "confidence": p[4]
                } for p in presidio_preds] if presidio_preds else [],
                "detection_count": {
                    "genetic": len(genetic_preds),
                    "presidio": len(presidio_preds) if presidio_preds else 0
                },
                "text_sample": text[:1000] + ("..." if len(text) > 1000 else "")
            }
            
            # Save detailed report to file
            file_base = os.path.splitext(file_info["file_name"])[0]
            report_path = os.path.join(output_dir, f"{file_base}_phi_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(file_report, f, indent=2)
        
        all_results.extend(batch_results)
    
    # Create summary DataFrame
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Save to CSV
        csv_path = os.path.join(output_dir, "phi_detection_results.csv")
        df.to_csv(csv_path, index=False)
        
        # Generate summary report
        summary = {
            "total_files": len(df),
            "total_phi_detected": {
                "genetic": df["genetic_count"].sum(),
                "presidio": df["presidio_count"].sum() if "presidio_count" in df.columns else 0
            },
            "files_with_phi": {
                "genetic": (df["genetic_count"] > 0).sum(),
                "presidio": (df["presidio_count"] > 0).sum() if "presidio_count" in df.columns else 0
            },
            "phi_types_found": {
                "genetic": list(set(typ for types in df["phi_types_genetic"] for typ in types)),
                "presidio": list(set(typ for types in df["phi_types_presidio"] for typ in types)) if "phi_types_presidio" in df.columns else []
            },
            "top_files_by_phi_count": df.sort_values("genetic_count", ascending=False)[["file_name", "genetic_count"]].head(10).to_dict(orient="records"),
            "detection_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_settings": {
                "genetic_generations": generations,
                "presidio_used": presidio_model is not None
            }
        }
        
        # Save summary
        summary_path = os.path.join(output_dir, "phi_detection_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nPHI Detection Summary:")
        print(f"Total files processed: {summary['total_files']}")
        print(f"Files with PHI detected (genetic): {summary['files_with_phi']['genetic']}")
        if presidio_model:
            print(f"Files with PHI detected (presidio): {summary['files_with_phi']['presidio']}")
        print(f"Total PHI instances found (genetic): {summary['total_phi_detected']['genetic']}")
        if presidio_model:
            print(f"Total PHI instances found (presidio): {summary['total_phi_detected']['presidio']}")
        print(f"\nPHI types found (genetic): {', '.join(summary['phi_types_found']['genetic'])}")
        print(f"\nResults saved to: {output_dir}")
        print(f"  - Detailed CSV: {os.path.basename(csv_path)}")
        print(f"  - Summary: {os.path.basename(summary_path)}")
        print(f"  - Individual reports: {file_base}_phi_report.json (for each file)")
        
        return df
    else:
        print("No valid results found.")
        return None


def main():
    """Parse arguments and process directory."""
    parser = argparse.ArgumentParser(description="Process any directory for PHI/PII detection")
    
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory to scan")
    parser.add_argument("--output", type=str, default="phi_detection_results",
                        help="Output directory for results")
    parser.add_argument("--generations", type=int, default=20,
                        help="Number of generations for genetic algorithm")
    parser.add_argument("--file_types", type=str, default=None,
                        help="Comma-separated list of file extensions to process (e.g. '.txt,.pdf,.docx')")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum number of files to process")
    parser.add_argument("--skip_presidio", action="store_true",
                        help="Skip using Presidio baseline")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Number of files to process in each batch")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Input directory not found: {args.input}")
        return 1
    
    # Parse file types if provided
    file_types = None
    if args.file_types:
        file_types = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                     for ext in args.file_types.split(',')]
        print(f"File types to process: {', '.join(file_types)}")
    
    # Scan directory for files
    file_paths = scan_directory(
        args.input, 
        file_types=file_types,
        max_files=args.max_files
    )
    
    if not file_paths:
        print("No files found matching the criteria.")
        return 1
    
    # Process files
    process_files(
        file_paths,
        args.output,
        generations=args.generations,
        use_presidio=not args.skip_presidio,
        batch_size=args.batch_size
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())