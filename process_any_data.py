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
import certifi
import sys
import argparse
import json
import numpy as np
import glob
import tempfile
import subprocess
import random
import shutil
import signal
import time
import traceback
import re
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
import pandas as pd

# Setup proper SSL verification for external requests
import ssl
import certifi

# Set environment variables and create proper SSL context
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Create a proper verified SSL context using certifi certificates
def create_verified_context():
    context = ssl.create_default_context(cafile=certifi.where())
    return context

# Replace the default context with our verified one
ssl._create_default_https_context = create_verified_context

# Configure basic logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("phi_processing.log")
    ]
)
logger = logging.getLogger("PHIDetector")

# Constants for safety
MAX_TEXT_SIZE = 1 * 1024 * 1024  # 1MB max text size
MAX_PDF_PAGES = 50  # Maximum PDF pages to process
PDF_TIMEOUT = 30  # Seconds per PDF file

# Define timeout exception
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
    logger.warning("Microsoft Presidio not available - will use fallback detection")


def extract_text_from_file(file_path, max_size=MAX_TEXT_SIZE):
    """
    Extract text from various file formats with detailed progress information.

    Args:
        file_path: Path to the file
        max_size: Maximum text size to extract

    Returns:
        Extracted text or empty string if extraction fails
    """
    start_time = time.time()
    file_path = str(file_path)
    file_name = os.path.basename(file_path)
    extension = os.path.splitext(file_path)[1].lower()

    print(f"\n{'='*70}")
    print(f"PROCESSING: {file_name}")
    print(f"File type: {extension}, Size: {os.path.getsize(file_path)/1024:.1f} KB")
    print(f"{'='*70}")

    # Plain text files
    if extension in ['.txt', '.csv', '.json', '.xml', '.html', '.md', '.log']:
        try:
            print(f"Reading text file: {file_name}")
            with time_limit(10):  # 10 second timeout for text files
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read(max_size)
                    duration = time.time() - start_time
                    print(f"✓ Text file read successfully: {len(text)} chars in {duration:.2f}s")
                    return text
        except TimeoutException:
            logger.error(f"⚠ Timeout reading text file: {file_name}")
            return ""
        except Exception as e:
            logger.error(f"⚠ Error reading {file_name}: {e}")
            return ""

    # PDF files
    elif extension == '.pdf':
        print(f"Extracting text from PDF: {file_name}")
        pdf_result = extract_text_from_pdf_safe(file_path, max_size)
        duration = time.time() - start_time
        print(f"✓ PDF extraction completed in {duration:.2f}s, extracted {len(pdf_result)} chars")
        return pdf_result

    # Microsoft Office documents
    elif extension in ['.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']:
        print(f"Extracting text from Office file: {file_name}")
        try:
            # Try textract if available
            try:
                import textract
                with time_limit(30):  # 30 second timeout for office docs
                    print("  Using textract for extraction...")
                    text = textract.process(file_path).decode('utf-8', errors='ignore')
                    text = text[:max_size]  # Limit size
                    duration = time.time() - start_time
                    print(f"✓ Office extraction completed in {duration:.2f}s, extracted {len(text)} chars")
                    return text
            except (ImportError, TimeoutException):
                print("  Textract failed or timed out, trying next method...")

            # Try Apache Tika if available
            try:
                import tika
                from tika import parser
                with time_limit(30):  # 30 second timeout
                    print("  Using Apache Tika for extraction...")
                    tika.initVM()
                    parsed = parser.from_file(file_path)
                    if parsed and 'content' in parsed and parsed['content']:
                        text = parsed['content'][:max_size]  # Limit size
                        duration = time.time() - start_time
                        print(f"✓ Office extraction completed in {duration:.2f}s, extracted {len(text)} chars")
                        return text
            except (ImportError, TimeoutException):
                print("  Apache Tika failed or timed out...")

            print("⚠ No suitable library for Office document extraction")
            return f"[OFFICE FILE: {file_name}] - Install textract or tika for content extraction"

        except Exception as e:
            logger.error(f"⚠ Error extracting from Office file {file_name}: {e}")
            logger.error(traceback.format_exc())
            return f"[OFFICE FILE: {file_name}] - Error during extraction"

    # Email files
    elif extension in ['.eml', '.msg']:
        print(f"Extracting text from email file: {file_name}")
        try:
            # Try email extraction if available
            import email
            with time_limit(10):  # 10 second timeout
                with open(file_path, 'rb') as f:
                    msg = email.message_from_binary_file(f)
                    text = ""
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            text += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    text = text[:max_size]  # Limit size
                    duration = time.time() - start_time
                    print(f"✓ Email extraction completed in {duration:.2f}s, extracted {len(text)} chars")
                    return text
        except ImportError:
            print("⚠ Email library not available")
            return f"[EMAIL FILE: {file_name}] - Install email library for content extraction"
        except TimeoutException:
            logger.error(f"⚠ Timeout extracting from email file: {file_name}")
            return ""
        except Exception as e:
            logger.error(f"⚠ Error extracting from email {file_name}: {e}")
            return f"[EMAIL FILE: {file_name}] - Error during extraction"

    # Default case for unsupported formats
    else:
        print(f"Attempting to read unknown format: {file_name}")
        # Try to read as text anyway for unknown formats
        try:
            with time_limit(5):  # 5 second timeout
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(max_size)
                    # Check if it looks like text
                    if '\0' not in content and len(content.strip()) > 0:
                        duration = time.time() - start_time
                        print(f"✓ Successfully read as text in {duration:.2f}s: {len(content)} chars")
                        return content
        except (TimeoutException, Exception):
            pass

        print(f"⚠ Unsupported file format: {file_name}")
        return f"[UNSUPPORTED FILE: {file_name}] - Format not supported for extraction"


def extract_text_from_pdf_safe(file_path, max_size=MAX_TEXT_SIZE):
    """
    Extract text from PDF files with safety measures to prevent hanging.

    Args:
        file_path: Path to the PDF file
        max_size: Maximum text size to extract

    Returns:
        Extracted text or empty string if extraction fails
    """
    file_name = os.path.basename(file_path)

    # First try the fastest method: pdftotext command line tool
    try:
        print(f"  Trying pdftotext command line tool...")
        with time_limit(15):  # 15 second timeout
            result = subprocess.run(
                ['pdftotext', file_path, '-'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=10
            )
            if result.returncode == 0 and result.stdout and len(result.stdout) > 100:
                text = result.stdout[:max_size]
                print(f"  ✓ pdftotext extraction successful: {len(text)} chars")
                return text
    except (TimeoutException, subprocess.TimeoutExpired, Exception) as e:
        print(f"  ⚠ pdftotext failed: {type(e).__name__}")

    # Try PyPDF2 with page-by-page extraction and timeouts
    try:
        print(f"  Trying PyPDF2 with page-by-page extraction...")
        import PyPDF2

        with open(file_path, 'rb') as pdf_file:
            # Creating reader with timeout
            try:
                with time_limit(10):
                    reader = PyPDF2.PdfReader(pdf_file)
                    num_pages = len(reader.pages)
                    print(f"  PDF has {num_pages} pages")

                    # Limit pages
                    max_pages = min(MAX_PDF_PAGES, num_pages)
                    if max_pages < num_pages:
                        print(f"  ⚠ Limiting to first {max_pages} pages (of {num_pages})")
            except (TimeoutException, Exception) as e:
                print(f"  ⚠ Error initializing PDF reader: {type(e).__name__}")
                return ""

            # Extract text page by page with timeout for each page
            text = ""
            pages_processed = 0

            for page_num in range(max_pages):
                print(f"  Processing page {page_num+1}/{max_pages}...", end="", flush=True)
                try:
                    with time_limit(5):  # 5 seconds per page maximum
                        page_text = reader.pages[page_num].extract_text() or ""
                        text += page_text + "\n"
                        pages_processed += 1

                        # Check if we've reached size limit
                        if len(text) >= max_size:
                            print(f" reached size limit")
                            print(f"  ⚠ Reached maximum text size at page {page_num+1}")
                            return text[:max_size]

                        print(f" done ({len(page_text)} chars)")
                except TimeoutException:
                    print(f" timed out")
                    print(f"  ⚠ Page {page_num+1} extraction timed out, skipping")
                    continue
                except Exception as e:
                    print(f" error ({type(e).__name__})")
                    print(f"  ⚠ Error extracting page {page_num+1}: {e}")
                    continue

            if pages_processed > 0 and len(text.strip()) > 100:
                print(f"  ✓ PyPDF2 extracted {pages_processed} pages: {len(text)} chars")
                return text[:max_size]

            print(f"  ⚠ PyPDF2 extraction insufficient: {pages_processed} pages, {len(text)} chars")

    except ImportError:
        print(f"  ⚠ PyPDF2 not available")
    except Exception as e:
        print(f"  ⚠ PyPDF2 extraction error: {type(e).__name__} - {e}")

    # Try pdfplumber as a last resort
    try:
        print(f"  Trying pdfplumber...")
        import pdfplumber

        with time_limit(PDF_TIMEOUT):
            with pdfplumber.open(file_path) as pdf:
                num_pages = len(pdf.pages)
                print(f"  PDF has {num_pages} pages (pdfplumber)")

                # Limit page count
                max_pages = min(MAX_PDF_PAGES, num_pages)
                if max_pages < num_pages:
                    print(f"  ⚠ Limiting to first {max_pages} pages (of {num_pages})")

                # Extract text page by page with timeout for each page
                text = ""
                pages_processed = 0

                for i in range(max_pages):
                    print(f"  Processing page {i+1}/{max_pages}...", end="", flush=True)
                    try:
                        with time_limit(5):  # 5 seconds per page maximum
                            page_text = pdf.pages[i].extract_text() or ""
                            text += page_text + "\n"
                            pages_processed += 1

                            # Check if we've reached size limit
                            if len(text) >= max_size:
                                print(f" reached size limit")
                                print(f"  ⚠ Reached maximum text size at page {i+1}")
                                return text[:max_size]

                            print(f" done ({len(page_text)} chars)")
                    except TimeoutException:
                        print(f" timed out")
                        print(f"  ⚠ Page {i+1} extraction timed out with pdfplumber")
                        continue
                    except Exception as e:
                        print(f" error ({type(e).__name__})")
                        print(f"  ⚠ Error extracting page {i+1} with pdfplumber: {e}")
                        continue

                if pages_processed > 0 and len(text.strip()) > 100:
                    print(f"  ✓ pdfplumber extracted {pages_processed} pages: {len(text)} chars")
                    return text[:max_size]

                print(f"  ⚠ pdfplumber extraction insufficient: {pages_processed} pages, {len(text)} chars")

    except ImportError:
        print(f"  ⚠ pdfplumber not available")
    except TimeoutException:
        print(f"  ⚠ pdfplumber timed out")
    except Exception as e:
        print(f"  ⚠ pdfplumber extraction error: {type(e).__name__} - {e}")

    # All methods failed
    print(f"⚠ All PDF extraction methods failed for {file_name}")
    return f"[PDF FILE: {file_name}] - PDF extraction failed. Not enough text extracted."


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


def process_files(file_paths, output_dir, generations=20, use_presidio=True, batch_size=5):
    """
    Process files for PHI detection with improved verbosity and safety.

    Args:
        file_paths: List of file paths to process
        output_dir: Directory to save results
        generations: Number of generations for genetic algorithm
        use_presidio: Whether to use Presidio baseline for comparison
        batch_size: Number of files to process in each batch (default: 5)

    Returns:
        DataFrame with results
    """
    print("\n" + "="*80)
    print(f"PHI DETECTION PROCESS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print(f"Total files to process: {len(file_paths)}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Genetic algorithm generations: {generations}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize models
    print("\nInitializing PHI detection models...")

    # Genetic algorithm model
    print("Initializing genetic algorithm model...")
    try:
        genetic_model = GeneticPIIDetector(
            population_size=50,
            generations=generations,
            crossover_prob=0.7,
            mutation_prob=0.3  # Increased mutation rate for better exploration
        )
        print("✓ Genetic algorithm model initialized")
    except Exception as e:
        print(f"⚠ Error initializing genetic model: {e}")
        print(traceback.format_exc())
        print("⚠ Exiting process - genetic model initialization failed")
        return None

    # Presidio baseline if available
    presidio_model = None
    if use_presidio and PRESIDIO_AVAILABLE:
        try:
            print("Initializing Presidio baseline...")
            presidio_model = PresidioBaseline()
            print("✓ Presidio baseline initialized successfully")
        except Exception as e:
            print(f"⚠ Error initializing Presidio: {e}")
            presidio_model = None

    # Process files in smaller batches
    all_results = []
    successful_files = 0
    failed_files = 0
    start_time = time.time()

    # Create regex patterns for fallback annotation generation
    patterns = [
        (re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'), "NAME"),  # Names
        (re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'), "PHONE"),  # Phone numbers
        (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), "EMAIL"),  # Emails
        (re.compile(r'\b\d{3}[-]?\d{2}[-]?\d{4}\b'), "SSN"),  # SSNs
        (re.compile(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'), "DATE"),  # Dates
        (re.compile(r'\bMRN:?\s*\d+\b'), "MEDICAL_RECORD"),  # Medical record numbers
    ]

    for batch_idx in range(0, len(file_paths), batch_size):
        batch_files = file_paths[batch_idx:batch_idx + batch_size]
        batch_start_time = time.time()

        print("\n" + "="*80)
        print(f"PROCESSING BATCH {batch_idx//batch_size + 1}/{(len(file_paths) + batch_size - 1)//batch_size}")
        print(f"Files in batch: {len(batch_files)}")
        print("="*80)

        # Extract text from each file - with more detailed reporting
        batch_texts = []
        extracted_info = []

        for i, file_path in enumerate(batch_files):
            try:
                print(f"\nFile {i+1}/{len(batch_files)} in batch")

                # Extract text with detailed information
                text = extract_text_from_file(file_path)

                # Skip files with no content
                if not text or len(text) < 50:
                    print(f"⚠ Insufficient text from {os.path.basename(file_path)}, skipping")
                    failed_files += 1
                    continue

                # Add to batch
                batch_texts.append(text)
                extracted_info.append({
                    "file_path": file_path,
                    "file_name": os.path.basename(file_path),
                    "file_size": os.path.getsize(file_path),
                    "text_length": len(text)
                })
                print(f"✓ Successfully extracted {len(text)} characters of text")

            except Exception as e:
                print(f"⚠ Error extracting text from {os.path.basename(file_path)}: {e}")
                print(traceback.format_exc())
                failed_files += 1

        if not batch_texts:
            print("⚠ No valid text extracted in this batch, skipping")
            continue

        # Print batch statistics
        print("\n" + "-"*80)
        print(f"BATCH STATISTICS")
        print(f"Files with extracted text: {len(batch_texts)}/{len(batch_files)}")
        total_chars = sum(len(t) for t in batch_texts)
        print(f"Total text: {total_chars} characters ({total_chars/1000:.1f}K)")
        print(f"Average text per file: {total_chars/len(batch_texts):.1f} characters")
        print("-"*80)

        # Generate synthetic annotations for training
        print("\nGenerating training annotations...")
        synthetic_annotations = []

        # Use Presidio or fallback to regex patterns
        if presidio_model:
            print("Using Presidio for annotation generation")
            for i, text in enumerate(batch_texts):
                try:
                    with time_limit(60):  # 1 minute timeout for annotation generation
                        print(f"  Generating annotations for file {i+1}/{len(batch_texts)}...", end="", flush=True)
                        presidio_results = presidio_model.detect(text)
                        annotations = [(start, end, pii_type) for _, start, end, pii_type, _ in presidio_results]
                        synthetic_annotations.append(annotations)
                        print(f" done ({len(annotations)} annotations)")
                except (TimeoutException, Exception) as e:
                    print(f" failed ({type(e).__name__})")
                    # Fallback to regex patterns
                    annotations = []
                    for pattern, pii_type in patterns:
                        for match in pattern.finditer(text):
                            annotations.append((match.start(), match.end(), pii_type))
                    synthetic_annotations.append(annotations)
                    print(f"  ↪ Fallback to regex patterns: {len(annotations)} annotations")
        else:
            # Use regex patterns for annotation
            print("Using regex patterns for annotation generation")
            for i, text in enumerate(batch_texts):
                try:
                    with time_limit(30):  # 30 seconds timeout
                        print(f"  Generating annotations for file {i+1}/{len(batch_texts)}...", end="", flush=True)
                        annotations = []
                        for pattern, pii_type in patterns:
                            for match in pattern.finditer(text):
                                annotations.append((match.start(), match.end(), pii_type))
                        synthetic_annotations.append(annotations)
                        print(f" done ({len(annotations)} annotations)")
                except (TimeoutException, Exception) as e:
                    print(f" failed ({type(e).__name__})")
                    # Create empty annotations as fallback
                    synthetic_annotations.append([])
                    print(f"  ↪ Using empty annotations as fallback")

        # Print annotation statistics
        total_annotations = sum(len(a) for a in synthetic_annotations)
        print(f"\nTotal annotations generated: {total_annotations}")
        print(f"Average annotations per file: {total_annotations/len(batch_texts):.1f}")

        # Train the genetic model with timeout
        print("\nTraining genetic algorithm...")
        try:
            with time_limit(300):  # 5 minute timeout for training
                print(f"Training for {generations} generations on {len(batch_texts)} documents...")
                genetic_model.train(batch_texts, synthetic_annotations)
                print("✓ Genetic algorithm training completed")
        except (TimeoutException, Exception) as e:
            print(f"⚠ Error during genetic algorithm training: {e}")
            print(traceback.format_exc())
            print("⚠ Continuing with potentially incomplete training")

        # Run detection for each file
        print("\nRunning PHI detection...")
        batch_results = []

        for i, text in enumerate(batch_texts):
            try:
                file_info = extracted_info[i]
                file_name = file_info["file_name"]
                print(f"\nDetecting PHI in file {i+1}/{len(batch_texts)}: {file_name}")

                # Genetic algorithm detection with timeout
                print("  Running genetic algorithm detection...", end="", flush=True)
                genetic_preds = []
                try:
                    with time_limit(60):  # 1 minute timeout for detection
                        genetic_preds = genetic_model.predict(text)
                        print(f" found {len(genetic_preds)} entities")
                except (TimeoutException, Exception) as e:
                    print(f" failed ({type(e).__name__})")
                    print(f"  ⚠ Error in genetic detection: {e}")

                # Presidio detection if available
                presidio_preds = []
                if presidio_model:
                    print("  Running Presidio detection...", end="", flush=True)
                    try:
                        with time_limit(60):  # 1 minute timeout for detection
                            presidio_preds = presidio_model.detect(text)
                            print(f" found {len(presidio_preds)} entities")
                    except (TimeoutException, Exception) as e:
                        print(f" failed ({type(e).__name__})")
                        print(f"  ⚠ Error in Presidio detection: {e}")

                # Regex detection (always run)
                regex_preds = []
                for pattern, pii_type in patterns:
                    for match in pattern.finditer(text):
                        matched_text = match.group()
                        start = match.start()
                        end = match.end()
                        regex_preds.append((matched_text, start, end, pii_type))
                print(f"  Regex detection found {len(regex_preds)} entities")

                # Record results
                result = {
                    "file_path": file_info["file_path"],
                    "file_name": file_info["file_name"],
                    "file_size": file_info["file_size"],
                    "text_length": file_info["text_length"],
                    "genetic_detections": genetic_preds,
                    "genetic_count": len(genetic_preds),
                    "presidio_detections": presidio_preds,
                    "presidio_count": len(presidio_preds),
                    "regex_detections": regex_preds,
                    "regex_count": len(regex_preds),
                    "phi_types_genetic": list(set(p[3] for p in genetic_preds)) if genetic_preds else [],
                    "phi_types_presidio": list(set(p[3] for p in presidio_preds)) if presidio_preds else [],
                    "phi_types_regex": list(set(p[3] for p in regex_preds)) if regex_preds else []
                }

                # Show PHI types found
                if genetic_preds:
                    print(f"  Genetic algorithm PHI types: {', '.join(result['phi_types_genetic'])}")
                if presidio_preds and presidio_model:
                    print(f"  Presidio PHI types: {', '.join(result['phi_types_presidio'])}")

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
                    "presidio_detections": [{
                        "text": p[0],
                        "start": p[1],
                        "end": p[2],
                        "type": p[3],
                        "confidence": float(p[4])
                    } for p in presidio_preds] if presidio_preds else [],
                    "regex_detections": [{
                        "text": p[0],
                        "start": p[1],
                        "end": p[2],
                        "type": p[3],
                        "confidence": 1.0
                    } for p in regex_preds] if regex_preds else [],
                    "detection_count": {
                        "genetic": len(genetic_preds),
                        "presidio": len(presidio_preds) if presidio_preds else 0,
                        "regex": len(regex_preds) if regex_preds else 0
                    },
                    "text_sample": text[:500] + ("..." if len(text) > 500 else "")
                }

                # Save detailed report to file
                file_base = os.path.splitext(file_info["file_name"])[0]
                report_path = os.path.join(output_dir, f"{file_base}_phi_report.json")

                try:
                    with open(report_path, "w", encoding="utf-8") as f:
                        # Convert numpy values to Python native types before serializing
                        if 'convert_numpy_types' not in locals():
                            def convert_numpy_types(obj):
                                if isinstance(obj, np.integer):
                                    return int(obj)
                                elif isinstance(obj, np.floating):
                                    return float(obj)
                                elif isinstance(obj, np.ndarray):
                                    return obj.tolist()
                                elif isinstance(obj, dict):
                                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                                elif isinstance(obj, list):
                                    return [convert_numpy_types(item) for item in obj]
                                return obj
                        file_report = convert_numpy_types(file_report)
                        json.dump(file_report, f, indent=2)
                    print(f"  ✓ Saved report to {os.path.basename(report_path)}")
                except Exception as e:
                    print(f"  ⚠ Error saving report: {e}")

                batch_results.append(result)
                successful_files += 1

            except Exception as e:
                print(f"⚠ Error processing file {i+1}: {e}")
                print(traceback.format_exc())
                failed_files += 1

        # Add batch results to overall results
        all_results.extend(batch_results)

        # Print batch summary
        batch_duration = time.time() - batch_start_time
        print("\n" + "-"*80)
        print(f"BATCH {batch_idx//batch_size + 1} COMPLETED")
        print(f"Time taken: {batch_duration:.1f} seconds ({batch_duration/60:.1f} minutes)")
        print(f"Files processed: {len(batch_results)}/{len(batch_files)}")
        print(f"Total PHI entities found (genetic): {sum(r['genetic_count'] for r in batch_results)}")
        if presidio_model:
            print(f"Total PHI entities found (presidio): {sum(r['presidio_count'] for r in batch_results)}")
        print("-"*80)

    # Create summary
    total_duration = time.time() - start_time
    print("\n" + "="*80)
    print(f"PHI DETECTION PROCESS COMPLETED")
    print(f"Total time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"Files processed: {successful_files}/{len(file_paths)} (failed: {failed_files})")
    print("="*80)

    if all_results:
        try:
            # Create summary DataFrame
            print("\nGenerating summary report...")
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj

            all_results_clean = [convert_numpy_types(r) for r in all_results]
            df = pd.DataFrame([{
                "file_path": r["file_path"],
                "file_name": r["file_name"],
                "file_size": r["file_size"],
                "text_length": r["text_length"],
                "genetic_count": r["genetic_count"],
                "presidio_count": r["presidio_count"] if "presidio_count" in r else 0,
                "regex_count": r["regex_count"] if "regex_count" in r else 0,
                "phi_types_genetic": ", ".join(r["phi_types_genetic"]) if "phi_types_genetic" in r else "",
                "phi_types_presidio": ", ".join(r["phi_types_presidio"]) if "phi_types_presidio" in r and r["phi_types_presidio"] else "",
                "phi_types_regex": ", ".join(r["phi_types_regex"]) if "phi_types_regex" in r and r["phi_types_regex"] else ""
            } for r in all_results_clean])

            # Save to CSV
            csv_path = os.path.join(output_dir, "phi_detection_results.csv")
            df.to_csv(csv_path, index=False)
            print(f"✓ Summary CSV saved to {os.path.basename(csv_path)}")

            # Generate summary report
            summary = {
                "total_files": len(df),
                "successful_files": successful_files,
                "failed_files": failed_files,
                "total_phi_detected": {
                    "genetic": df["genetic_count"].sum(),
                    "presidio": df["presidio_count"].sum() if "presidio_count" in df.columns else 0
                },
                "files_with_phi": {
                    "genetic": (df["genetic_count"] > 0).sum(),
                    "presidio": (df["presidio_count"] > 0).sum() if "presidio_count" in df.columns else 0
                },
                "phi_types_found": {
                    "genetic": list(set(typ.strip() for types_str in df["phi_types_genetic"]
                                        for typ in types_str.split(",") if typ.strip())),
                    "presidio": list(set(typ.strip() for types_str in df["phi_types_presidio"]
                                         for typ in types_str.split(",") if typ.strip())) if "phi_types_presidio" in df.columns else []
                },
                "top_files_by_phi_count": df.sort_values("genetic_count", ascending=False)[["file_name", "genetic_count"]].head(10).to_dict(orient="records"),
                "detection_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "processing_time_seconds": total_duration,
                "model_settings": {
                    "genetic_generations": generations,
                    "presidio_used": presidio_model is not None
                }
            }

            # Save summary
            summary_path = os.path.join(output_dir, "phi_detection_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                summary = convert_numpy_types(summary)
                json.dump(summary, f, indent=2)
            print(f"✓ Summary report saved to {os.path.basename(summary_path)}")

            # Print summary info
            print("\nPHI DETECTION SUMMARY:")
            print(f"Total files processed: {summary['total_files']} (success: {summary['successful_files']}, failed: {summary['failed_files']})")
            print(f"Files with PHI detected (genetic): {summary['files_with_phi']['genetic']}")
            if presidio_model:
                print(f"Files with PHI detected (presidio): {summary['files_with_phi']['presidio']}")
            print(f"Total PHI instances found (genetic): {summary['total_phi_detected']['genetic']}")
            if presidio_model:
                print(f"Total PHI instances found (presidio): {summary['total_phi_detected']['presidio']}")

            if summary['phi_types_found']['genetic']:
                print(f"\nPHI types found (genetic): {', '.join(summary['phi_types_found']['genetic'])}")

            print(f"\nResults saved to: {output_dir}")

            return df

        except Exception as e:
            print(f"⚠ Error generating summary: {e}")
            print(traceback.format_exc())
            return None
    else:
        print("No valid results found.")
        return None


def main():
    """Parse arguments and process directory."""
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
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Number of files to process in each batch (default: 5)")
    parser.add_argument("--skip_pdfs", action="store_true",
                        help="Skip processing PDF files")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose debug output")

    args = parser.parse_args()

    if args.debug:
        print("\nRUNNING IN DEBUG MODE")
        # Set logging level to DEBUG
        logging.getLogger().setLevel(logging.DEBUG)
        # Print all arguments
        print("Command-line arguments:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")

    print(f"\nStarting PHI detection on {args.input}")
    print(f"Results will be saved to {args.output}")

    if not os.path.exists(args.input):
        print(f"Input directory not found: {args.input}")
        return 1

    # Parse file types if provided
    file_types = None
    if args.file_types:
        file_types = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
                     for ext in args.file_types.split(',')]
        print(f"File types to process: {', '.join(file_types)}")
    elif args.skip_pdfs:
        # If skipping PDFs but no specific file types provided, process all except PDFs
        file_types = ['.txt', '.csv', '.json', '.xml', '.html', '.md', '.log', '.docx', '.doc']
        print(f"Skipping PDFs, processing only: {', '.join(file_types)}")

    # Scan directory for files
    print(f"Scanning directory: {args.input}")
    file_paths = scan_directory(
        args.input,
        file_types=file_types,
        max_files=args.max_files
    )

    if not file_paths:
        print("No files found matching the criteria.")
        return 1

    # Print stats about found files
    print("\nFile statistics:")
    extensions = {}
    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()
        extensions[ext] = extensions.get(ext, 0) + 1

    for ext, count in sorted(extensions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ext or 'no extension'}: {count} files")

    # Process files with configured batch size
    try:
        process_files(
            file_paths,
            args.output,
            generations=args.generations,
            use_presidio=not args.skip_presidio,
            batch_size=args.batch_size
        )
        print(f"\nProcess completed successfully. Results saved to {args.output}/")
        return 0
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        print("Partial results may have been saved.")
        return 130
    except Exception as e:
        print(f"\n\nError during processing: {e}")
        print(traceback.format_exc())
        print("\nThe process encountered an error. Partial results may have been saved.")
        return 1


if __name__ == "__main__":
    sys.exit(main())