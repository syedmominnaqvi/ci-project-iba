#!/usr/bin/env python
"""
Robust process_any_data script for PHI/PII detection.
This version includes:
1. Improved error handling with timeouts
2. Memory-efficient batch processing
3. Robust file handling for large datasets
4. Detailed logging for debugging
5. Graceful recovery from processing errors
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
import traceback
import signal
import time
import logging
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("phi_processing.log")
    ]
)
logger = logging.getLogger("PHIDetector")

# Constants for safety limits
MAX_TEXT_SIZE = 2 * 1024 * 1024  # 2MB max text size to process
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB max file size
MAX_PROCESSING_TIME = 300  # 5 minutes max processing time per file
DEFAULT_BATCH_SIZE = 5  # Process fewer files per batch

# Check for optional dependencies
TQDM_AVAILABLE = False
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    # Define a simple replacement if tqdm is not available
    def tqdm(iterable, **kwargs):
        if "desc" in kwargs:
            logger.info(f"{kwargs['desc']}...")
        return iterable

# Import our PHI detection system
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from src.genetic_algorithm import GeneticPIIDetector
    from src.chromosome import DetectionGene, Chromosome
    GENETIC_AVAILABLE = True
except ImportError:
    logger.warning("Genetic algorithm module not available, using fallback detection")
    GENETIC_AVAILABLE = False

try:
    from src.baseline import PresidioBaseline
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    logger.warning("Microsoft Presidio not available")


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


def extract_text_from_file(file_path, max_size=MAX_TEXT_SIZE):
    """
    Extract text from various file formats with better error handling.
    
    Args:
        file_path: Path to the file
        max_size: Maximum text size to extract
        
    Returns:
        Extracted text or empty string if extraction fails
    """
    file_path = str(file_path)
    extension = os.path.splitext(file_path)[1].lower()
    
    # Check file size first to avoid processing very large files
    try:
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE:
            logger.warning(f"File too large to process: {file_path} ({file_size / (1024*1024):.1f} MB)")
            return ""
    except Exception as e:
        logger.error(f"Error checking file size for {file_path}: {e}")
        return ""
    
    # Plain text files
    if extension in ['.txt', '.csv', '.json', '.xml', '.html', '.md', '.log']:
        try:
            with time_limit(30):  # 30 seconds timeout for text file reading
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read(max_size)
                    if len(text) >= max_size:
                        logger.warning(f"Truncated {file_path} to {max_size} characters")
                    return text
        except TimeoutException:
            logger.error(f"Timeout reading {file_path}")
            return ""
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return ""
    
    # PDF files
    elif extension == '.pdf':
        try:
            # Try using PyPDF2 which doesn't require external tools
            try:
                with time_limit(120):  # 2 minutes timeout for PDF processing
                    import PyPDF2
                    with open(file_path, 'rb') as pdf_file:
                        reader = PyPDF2.PdfReader(pdf_file)
                        # Limit to 50 pages maximum
                        max_pages = min(50, len(reader.pages))
                        text = ""
                        for page_num in range(max_pages):
                            text += reader.pages[page_num].extract_text() or ""
                            # Check size limit during extraction
                            if len(text) >= max_size:
                                logger.warning(f"PDF extraction reached size limit for {file_path}")
                                return text[:max_size]
                        if text.strip():
                            return text
            except (ImportError, TimeoutException) as e:
                logger.warning(f"PyPDF2 extraction failed or timed out: {e}")
            except Exception as e:
                logger.warning(f"PyPDF2 error: {e}")

            # Try using pdfplumber which has better extraction
            try:
                with time_limit(120):  # 2 minutes timeout for PDF processing
                    import pdfplumber
                    with pdfplumber.open(file_path) as pdf:
                        # Limit to 50 pages maximum
                        max_pages = min(50, len(pdf.pages))
                        text = ""
                        for i, page in enumerate(pdf.pages[:max_pages]):
                            text += page.extract_text() or ""
                            # Check size limit during extraction
                            if len(text) >= max_size:
                                logger.warning(f"PDF extraction reached size limit for {file_path}")
                                return text[:max_size]
                        if text.strip():
                            return text
            except (ImportError, TimeoutException) as e:
                logger.warning(f"pdfplumber extraction failed or timed out: {e}")
            except Exception as e:
                logger.warning(f"pdfplumber error: {e}")

            # Try command-line tools as last resort
            try:
                with time_limit(60):  # 1 minute timeout for command-line tools
                    # Try pdftotext if available
                    result = subprocess.run(
                        ['pdftotext', file_path, '-'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True,
                        timeout=30
                    )
                    if result.returncode == 0 and result.stdout:
                        return result.stdout[:max_size]
            except (TimeoutException, subprocess.TimeoutExpired):
                logger.warning(f"Command-line PDF extraction timed out for {file_path}")
            except Exception as e:
                logger.warning(f"Command-line extraction error: {e}")

            # Return placeholder if all methods failed
            logger.error(f"All PDF extraction methods failed for {file_path}")
            return ""

        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            return ""
    
    # Microsoft Office documents
    elif extension in ['.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']:
        try:
            with time_limit(120):  # 2 minutes timeout for office document processing
                # Try textract if available
                try:
                    import textract
                    text = textract.process(file_path).decode('utf-8', errors='ignore')
                    return text[:max_size]
                except ImportError:
                    logger.warning("textract not available for Office file extraction")
                except Exception as e:
                    logger.warning(f"textract extraction error: {e}")
                
                # Try Apache Tika if available
                try:
                    import tika
                    from tika import parser
                    tika.initVM()
                    parsed = parser.from_file(file_path)
                    if parsed and 'content' in parsed and parsed['content']:
                        return parsed['content'][:max_size]
                except ImportError:
                    logger.warning("tika not available for Office file extraction")
                except Exception as e:
                    logger.warning(f"tika extraction error: {e}")
            
            # If we get here, both methods failed
            logger.error(f"Office file extraction failed for {file_path}")
            return ""
            
        except TimeoutException:
            logger.error(f"Timeout extracting text from Office file {file_path}")
            return ""
        except Exception as e:
            logger.error(f"Error extracting text from Office file {file_path}: {e}")
            return ""
    
    # Default case for unsupported formats
    else:
        # Try to read as text anyway for unknown formats
        try:
            with time_limit(30):  # 30 seconds timeout
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(max_size)
                    # Check if it looks like text
                    if '\0' not in content and len(content.strip()) > 0:
                        return content
        except (TimeoutException, Exception):
            pass
        
        logger.warning(f"Unsupported file format: {file_path}")
        return ""


def scan_directory(input_dir, file_types=None, max_files=None, min_size=10, max_size=MAX_FILE_SIZE):
    """
    Recursively scan directory for files to process with improved error handling.
    
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
        logger.error(f"Directory not found: {input_dir}")
        return []
    
    logger.info(f"Scanning directory: {input_dir}")
    
    all_files = []
    skipped_files = 0
    
    try:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                try:
                    file_path = os.path.join(root, file)
                    
                    # Check file size
                    try:
                        file_size = os.path.getsize(file_path)
                        if file_size < min_size or file_size > max_size:
                            skipped_files += 1
                            continue
                    except Exception as e:
                        logger.debug(f"Error checking file size for {file_path}: {e}")
                        skipped_files += 1
                        continue
                    
                    # Check file extension if specified
                    if file_types:
                        ext = os.path.splitext(file)[1].lower()
                        if ext not in file_types:
                            skipped_files += 1
                            continue
                    
                    all_files.append(file_path)
                    
                    # Add progress indicator for large directories
                    if len(all_files) % 1000 == 0:
                        logger.info(f"Found {len(all_files)} files so far...")
                        
                except Exception as e:
                    logger.debug(f"Error processing file {file} during scan: {e}")
                    skipped_files += 1
    except Exception as e:
        logger.error(f"Error scanning directory: {e}")
    
    logger.info(f"Found {len(all_files)} files matching criteria (skipped {skipped_files} files)")
    
    # Limit number of files if specified
    if max_files and len(all_files) > max_files:
        logger.info(f"Limiting to {max_files} files (randomly selected)")
        random.shuffle(all_files)
        all_files = all_files[:max_files]
    
    return all_files


def create_baseline_detector():
    """
    Create a simple regex-based detector as fallback.
    """
    try:
        # Use our own implementation if available
        from src.chromosome import DetectionGene, Chromosome
        
        genes = [
            DetectionGene(pattern=r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', pii_type="NAME", confidence=0.8),
            DetectionGene(pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', pii_type="EMAIL", confidence=0.9),
            DetectionGene(pattern=r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', pii_type="PHONE", confidence=0.8),
            DetectionGene(pattern=r'\b\d{3}[-]?\d{2}[-]?\d{4}\b', pii_type="SSN", confidence=0.9),
            DetectionGene(pattern=r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', pii_type="DATE", confidence=0.7),
            DetectionGene(pattern=r'\b\d+\s+[A-Z][a-z]+\s+[A-Za-z]+\b', pii_type="ADDRESS", confidence=0.7),
            DetectionGene(pattern=r'\bMRN:?\s*\d+\b', pii_type="MEDICAL_RECORD", confidence=0.9),
            DetectionGene(pattern=r'\bPatient ID:?\s*\d+\b', pii_type="MEDICAL_RECORD", confidence=0.9),
        ]
        return Chromosome(genes=genes)
    except Exception as e:
        logger.error(f"Error creating baseline detector: {e}")
        
        # Simple implementation as fallback
        import re
        
        class SimpleDetector:
            def __init__(self):
                self.patterns = [
                    (r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', "NAME", 0.8),
                    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "EMAIL", 0.9),
                    (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', "PHONE", 0.8),
                    (r'\b\d{3}[-]?\d{2}[-]?\d{4}\b', "SSN", 0.9),
                    (r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', "DATE", 0.7),
                    (r'\b\d+\s+[A-Z][a-z]+\s+[A-Za-z]+\b', "ADDRESS", 0.7),
                ]
                # Pre-compile patterns for efficiency
                self.compiled = [(re.compile(p), t, c) for p, t, c in self.patterns]
            
            def detect(self, text):
                results = []
                for pattern, pii_type, confidence in self.compiled:
                    for match in pattern.finditer(text):
                        start, end = match.span()
                        results.append((match.group(), start, end, pii_type, confidence))
                return results
        
        return SimpleDetector()


def safe_detect(model, text, timeout=MAX_PROCESSING_TIME):
    """
    Safely apply detection with timeout to avoid hanging on problematic files.
    """
    try:
        with time_limit(timeout):
            if hasattr(model, 'predict'):
                return model.predict(text)
            else:
                return model.detect(text)
    except TimeoutException:
        logger.warning(f"Detection timed out after {timeout} seconds")
        return []
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        return []


def process_files(file_paths, output_dir, generations=20, use_presidio=True, batch_size=DEFAULT_BATCH_SIZE):
    """
    Process files for PHI detection with robust error handling.
    
    Args:
        file_paths: List of file paths to process
        output_dir: Directory to save results
        generations: Number of generations for genetic algorithm
        use_presidio: Whether to use Presidio baseline for comparison
        batch_size: Number of files to process in each batch
        
    Returns:
        Dictionary with results summary
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a unique run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting PHI detection run {run_id}")
    
    # Initialize models
    logger.info("Initializing PHI detection models...")
    
    # Genetic algorithm model
    genetic_model = None
    if GENETIC_AVAILABLE:
        try:
            genetic_model = GeneticPIIDetector(
                population_size=50,
                generations=generations,
                crossover_prob=0.7,
                mutation_prob=0.3
            )
            logger.info("Genetic algorithm model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing genetic model: {e}")
            genetic_model = None
    
    # Baseline detector - always create this as fallback
    baseline_detector = create_baseline_detector()
    logger.info("Baseline regex detector initialized")
    
    # Presidio baseline if available and requested
    presidio_model = None
    if use_presidio and PRESIDIO_AVAILABLE:
        try:
            presidio_model = PresidioBaseline()
            logger.info("Presidio baseline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Presidio: {e}")
            presidio_model = None
    
    # Process files in batches
    all_results = []
    successful_files = 0
    failed_files = 0
    
    # Keep track of detection stats
    detection_stats = {
        "genetic": {"total": 0, "by_type": {}},
        "baseline": {"total": 0, "by_type": {}},
        "presidio": {"total": 0, "by_type": {}}
    }
    
    # Process in smaller batches
    for batch_idx in range(0, len(file_paths), batch_size):
        batch_files = file_paths[batch_idx:batch_idx + batch_size]
        
        logger.info(f"\nProcessing batch {batch_idx//batch_size + 1}/{(len(file_paths) + batch_size - 1)//batch_size}")
        
        # Extract text from each file
        batch_texts = []
        extracted_info = []
        
        for file_path in tqdm(batch_files, desc="Extracting text"):
            try:
                logger.debug(f"Extracting text from {file_path}")
                text = extract_text_from_file(file_path)
                
                # Skip files with no content
                if not text or len(text) < 50:
                    logger.debug(f"Insufficient text extracted from {file_path}")
                    failed_files += 1
                    continue
                
                batch_texts.append(text)
                extracted_info.append({
                    "file_path": file_path,
                    "file_name": os.path.basename(file_path),
                    "file_size": os.path.getsize(file_path),
                    "text_length": len(text)
                })
                
            except Exception as e:
                logger.error(f"Error extracting text from {file_path}: {e}")
                failed_files += 1
        
        if not batch_texts:
            logger.warning("No valid text extracted in this batch, skipping")
            continue
        
        logger.info(f"Successfully extracted text from {len(batch_texts)} files")
        logger.info(f"Total text size: {sum(len(t) for t in batch_texts)/1000:.1f}K characters")
        
        # Generate synthetic annotations for training
        synthetic_annotations = []
        
        try:
            # Use Presidio to generate initial annotations if available
            if presidio_model:
                logger.info("Generating training annotations with Presidio...")
                for text in tqdm(batch_texts, desc="Generating training annotations"):
                    try:
                        with time_limit(60):  # 1 minute timeout per file
                            presidio_results = presidio_model.detect(text)
                            annotations = [(start, end, pii_type) for _, start, end, pii_type, _ in presidio_results]
                            synthetic_annotations.append(annotations)
                    except (TimeoutException, Exception) as e:
                        logger.warning(f"Error generating annotations: {e}")
                        # Add empty annotations as fallback
                        synthetic_annotations.append([])
            else:
                # Use baseline detector for annotations
                logger.info("Generating training annotations with baseline detector...")
                for text in tqdm(batch_texts, desc="Generating training annotations"):
                    try:
                        with time_limit(60):  # 1 minute timeout per file
                            baseline_results = baseline_detector.detect(text)
                            annotations = [(start, end, pii_type) for _, start, end, pii_type, _ in baseline_results]
                            synthetic_annotations.append(annotations)
                    except (TimeoutException, Exception) as e:
                        logger.warning(f"Error generating annotations: {e}")
                        # Add empty annotations as fallback
                        synthetic_annotations.append([])
        except Exception as e:
            logger.error(f"Error generating annotations: {e}")
            # Create empty annotations if all else fails
            synthetic_annotations = [[] for _ in batch_texts]
        
        # Train the genetic model if available
        if genetic_model:
            try:
                logger.info(f"Training genetic algorithm for {generations} generations...")
                with time_limit(MAX_PROCESSING_TIME * 2):  # Allow more time for training
                    genetic_model.train(batch_texts, synthetic_annotations)
                logger.info("Genetic algorithm training completed")
            except (TimeoutException, Exception) as e:
                logger.error(f"Error training genetic model: {e}")
                # If training fails, genetic_model will still be defined but may not work well
        
        # Run detection on each file
        batch_results = []
        
        for i, text in enumerate(tqdm(batch_texts, desc="Detecting PHI")):
            try:
                file_info = extracted_info[i]
                logger.debug(f"Processing {file_info['file_name']}")
                
                file_result = {
                    "file_path": file_info["file_path"],
                    "file_name": file_info["file_name"],
                    "file_size": file_info["file_size"],
                    "text_length": file_info["text_length"],
                }
                
                # Genetic algorithm detection
                genetic_preds = []
                if genetic_model:
                    try:
                        genetic_preds = safe_detect(genetic_model, text)
                        logger.debug(f"Genetic model found {len(genetic_preds)} entities")
                    except Exception as e:
                        logger.error(f"Error in genetic detection: {e}")
                
                # Baseline detection
                baseline_preds = []
                try:
                    baseline_preds = safe_detect(baseline_detector, text)
                    logger.debug(f"Baseline detector found {len(baseline_preds)} entities")
                except Exception as e:
                    logger.error(f"Error in baseline detection: {e}")
                
                # Presidio detection if available
                presidio_preds = []
                if presidio_model:
                    try:
                        presidio_preds = safe_detect(presidio_model, text)
                        logger.debug(f"Presidio found {len(presidio_preds)} entities")
                    except Exception as e:
                        logger.error(f"Error in Presidio detection: {e}")
                
                # Update detection stats
                detection_stats["genetic"]["total"] += len(genetic_preds)
                detection_stats["baseline"]["total"] += len(baseline_preds)
                detection_stats["presidio"]["total"] += len(presidio_preds)
                
                # Update type counts
                for _, _, _, pii_type, _ in genetic_preds:
                    detection_stats["genetic"]["by_type"][pii_type] = detection_stats["genetic"]["by_type"].get(pii_type, 0) + 1
                
                for _, _, _, pii_type, _ in baseline_preds:
                    detection_stats["baseline"]["by_type"][pii_type] = detection_stats["baseline"]["by_type"].get(pii_type, 0) + 1
                
                for _, _, _, pii_type, _ in presidio_preds:
                    detection_stats["presidio"]["by_type"][pii_type] = detection_stats["presidio"]["by_type"].get(pii_type, 0) + 1
                
                # Record results
                file_result.update({
                    "genetic_count": len(genetic_preds),
                    "baseline_count": len(baseline_preds),
                    "presidio_count": len(presidio_preds) if presidio_model else 0,
                    "phi_types_genetic": list(set(p[3] for p in genetic_preds)),
                    "phi_types_baseline": list(set(p[3] for p in baseline_preds)),
                    "phi_types_presidio": list(set(p[3] for p in presidio_preds)) if presidio_model else []
                })
                
                # Generate a detailed report for this file
                file_report = {
                    "file_info": file_info,
                    "genetic_detections": [{
                        "text": p[0],
                        "start": p[1],
                        "end": p[2],
                        "type": p[3],
                        "confidence": float(p[4])
                    } for p in genetic_preds],
                    "baseline_detections": [{
                        "text": p[0],
                        "start": p[1],
                        "end": p[2],
                        "type": p[3],
                        "confidence": float(p[4])
                    } for p in baseline_preds],
                    "presidio_detections": [{
                        "text": p[0],
                        "start": p[1],
                        "end": p[2],
                        "type": p[3],
                        "confidence": float(p[4])
                    } for p in presidio_preds] if presidio_model else [],
                    "detection_count": {
                        "genetic": len(genetic_preds),
                        "baseline": len(baseline_preds),
                        "presidio": len(presidio_preds) if presidio_model else 0
                    }
                }
                
                # Save detailed report to file
                file_base = os.path.splitext(file_info["file_name"])[0]
                report_path = os.path.join(output_dir, f"{file_base}_phi_report.json")
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(file_report, f, indent=2, default=str)
                
                batch_results.append(file_result)
                successful_files += 1
                
            except Exception as e:
                logger.error(f"Error processing file #{i+1}: {e}")
                failed_files += 1
        
        all_results.extend(batch_results)
        logger.info(f"Completed batch with {len(batch_results)} successful files")
    
    # Create summary
    try:
        if all_results:
            # Try to use pandas for summary if available
            try:
                import pandas as pd
                df = pd.DataFrame(all_results)
                
                # Save to CSV
                csv_path = os.path.join(output_dir, "phi_detection_results.csv")
                df.to_csv(csv_path, index=False)
                logger.info(f"Results saved to CSV: {csv_path}")
                
                # Generate summary statistics
                files_with_phi_genetic = (df["genetic_count"] > 0).sum() if "genetic_count" in df.columns else 0
                files_with_phi_baseline = (df["baseline_count"] > 0).sum() if "baseline_count" in df.columns else 0
                files_with_phi_presidio = (df["presidio_count"] > 0).sum() if "presidio_count" in df.columns else 0
                
                total_genetic = df["genetic_count"].sum() if "genetic_count" in df.columns else 0
                total_baseline = df["baseline_count"].sum() if "baseline_count" in df.columns else 0
                total_presidio = df["presidio_count"].sum() if "presidio_count" in df.columns else 0
                
                # Top files by PHI count
                top_files = df.sort_values("genetic_count" if "genetic_count" in df.columns else "baseline_count", 
                                         ascending=False)[["file_name", "genetic_count" if "genetic_count" in df.columns else "baseline_count"]].head(10).to_dict(orient="records")
            
            except (ImportError, Exception) as e:
                logger.warning(f"Error using pandas: {e}")
                # Fallback to manual calculations
                files_with_phi_genetic = sum(1 for r in all_results if r.get("genetic_count", 0) > 0)
                files_with_phi_baseline = sum(1 for r in all_results if r.get("baseline_count", 0) > 0)
                files_with_phi_presidio = sum(1 for r in all_results if r.get("presidio_count", 0) > 0)
                
                total_genetic = sum(r.get("genetic_count", 0) for r in all_results)
                total_baseline = sum(r.get("baseline_count", 0) for r in all_results)
                total_presidio = sum(r.get("presidio_count", 0) for r in all_results)
                
                # Top files by PHI count
                sorted_results = sorted(all_results, key=lambda x: x.get("genetic_count", 0) or x.get("baseline_count", 0), 
                                      reverse=True)
                top_files = [{"file_name": r["file_name"], 
                              "count": r.get("genetic_count", 0) or r.get("baseline_count", 0)} 
                             for r in sorted_results[:10]]
            
            # Generate summary
            phi_types_genetic = list(detection_stats["genetic"]["by_type"].keys())
            phi_types_baseline = list(detection_stats["baseline"]["by_type"].keys())
            phi_types_presidio = list(detection_stats["presidio"]["by_type"].keys())
            
            summary = {
                "run_id": run_id,
                "total_files_processed": successful_files + failed_files,
                "successful_files": successful_files,
                "failed_files": failed_files,
                "total_phi_detected": {
                    "genetic": total_genetic,
                    "baseline": total_baseline,
                    "presidio": total_presidio
                },
                "files_with_phi": {
                    "genetic": files_with_phi_genetic,
                    "baseline": files_with_phi_baseline,
                    "presidio": files_with_phi_presidio
                },
                "phi_types_found": {
                    "genetic": phi_types_genetic,
                    "baseline": phi_types_baseline,
                    "presidio": phi_types_presidio
                },
                "phi_counts_by_type": {
                    "genetic": detection_stats["genetic"]["by_type"],
                    "baseline": detection_stats["baseline"]["by_type"],
                    "presidio": detection_stats["presidio"]["by_type"]
                },
                "top_files_by_phi_count": top_files,
                "detection_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_settings": {
                    "genetic_available": GENETIC_AVAILABLE,
                    "genetic_generations": generations,
                    "presidio_used": presidio_model is not None
                }
            }
            
            # Save summary
            summary_path = os.path.join(output_dir, "phi_detection_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"\n{'='*50}")
            logger.info("PHI Detection Summary:")
            logger.info(f"{'='*50}")
            logger.info(f"Total files processed: {summary['total_files_processed']} ({summary['successful_files']} successful, {summary['failed_files']} failed)")
            
            if GENETIC_AVAILABLE:
                logger.info(f"Files with PHI detected (genetic): {summary['files_with_phi']['genetic']}")
                logger.info(f"Total PHI instances found (genetic): {summary['total_phi_detected']['genetic']}")
                logger.info(f"PHI types found (genetic): {', '.join(summary['phi_types_found']['genetic'])}")
            
            logger.info(f"Files with PHI detected (baseline): {summary['files_with_phi']['baseline']}")
            logger.info(f"Total PHI instances found (baseline): {summary['total_phi_detected']['baseline']}")
            logger.info(f"PHI types found (baseline): {', '.join(summary['phi_types_found']['baseline'])}")
            
            if presidio_model:
                logger.info(f"Files with PHI detected (presidio): {summary['files_with_phi']['presidio']}")
                logger.info(f"Total PHI instances found (presidio): {summary['total_phi_detected']['presidio']}")
                logger.info(f"PHI types found (presidio): {', '.join(summary['phi_types_found']['presidio'])}")
            
            logger.info(f"\nResults saved to: {output_dir}")
            logger.info(f"  - Summary: {os.path.basename(summary_path)}")
            logger.info(f"  - Individual reports: {len(all_results)} files")
            
            return summary
        else:
            logger.warning("No valid results found.")
            return None
            
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        logger.error(traceback.format_exc())
        return {
            "error": str(e),
            "total_files_processed": len(file_paths),
            "successful_files": successful_files,
            "failed_files": failed_files
        }


def main():
    """Parse arguments and process directory with robust error handling."""
    parser = argparse.ArgumentParser(description="Process any directory for PHI/PII detection")
    
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory to scan")
    parser.add_argument("--output", type=str, default="phi_results",
                        help="Output directory for results")
    parser.add_argument("--generations", type=int, default=20,
                        help="Number of generations for genetic algorithm")
    parser.add_argument("--file_types", type=str, default=None,
                        help="Comma-separated list of file extensions to process (e.g. '.txt,.pdf,.docx')")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum number of files to process")
    parser.add_argument("--skip_presidio", action="store_true",
                        help="Skip using Presidio baseline")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Number of files to process in each batch")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    
    try:
        args = parser.parse_args()
        
        # Set logging level based on debug flag
        if args.debug:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"Starting PHI detection process")
        logger.info(f"Input directory: {args.input}")
        logger.info(f"Output directory: {args.output}")
        logger.info(f"Genetic algorithm generations: {args.generations}")
        logger.info(f"Batch size: {args.batch_size}")
        
        if not os.path.exists(args.input):
            logger.error(f"Input directory not found: {args.input}")
            return 1
        
        # Parse file types if provided
        file_types = None
        if args.file_types:
            file_types = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                         for ext in args.file_types.split(',')]
            logger.info(f"File types to process: {', '.join(file_types)}")
        
        # Scan directory for files
        file_paths = scan_directory(
            args.input, 
            file_types=file_types,
            max_files=args.max_files
        )
        
        if not file_paths:
            logger.error("No files found matching the criteria.")
            return 1
        
        # Process files
        process_files(
            file_paths,
            args.output,
            generations=args.generations,
            use_presidio=not args.skip_presidio,
            batch_size=args.batch_size
        )
        
        logger.info("PHI detection completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())