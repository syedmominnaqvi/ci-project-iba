#!/usr/bin/env python
"""
Drop-in replacement for process_any_data.py's extract_text_from_file function.
Focuses specifically on fixing the PDF extraction issues causing the script to hang.
"""
import os
import sys
import time
import signal
import logging
import subprocess
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pdf_extraction.log")
    ]
)
logger = logging.getLogger("PDFExtractor")

# Constants
MAX_TEXT_SIZE = 1 * 1024 * 1024  # 1MB max text size from a PDF
MAX_PDF_PROCESSING_TIME = 30  # 30 seconds timeout for any single PDF
MAX_PAGE_COUNT = 50  # Maximum pages to extract from a PDF


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


def extract_text_from_pdf_safe(file_path, max_size=MAX_TEXT_SIZE):
    """
    Extract text from PDF files with aggressive timeout protection.
    This function tries multiple methods with strict timeouts to avoid hanging.
    
    Args:
        file_path: Path to the PDF file
        max_size: Maximum text size to extract in bytes
        
    Returns:
        Extracted text or empty string if extraction fails
    """
    logger.info(f"Extracting text from PDF: {os.path.basename(file_path)}")
    
    # Method 1: Simple pdftotext command-line tool (fastest)
    try:
        with time_limit(MAX_PDF_PROCESSING_TIME // 3):  # Allocate 1/3 of total time
            logger.debug("Trying pdftotext command line tool")
            result = subprocess.run(
                ['pdftotext', file_path, '-'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=10
            )
            if result.returncode == 0 and result.stdout and len(result.stdout) > 100:
                logger.debug("pdftotext extraction successful")
                return result.stdout[:max_size]
    except (TimeoutException, subprocess.TimeoutExpired):
        logger.warning("pdftotext timed out")
    except Exception as e:
        logger.debug(f"pdftotext failed: {e}")
    
    # Method 2: Use PyPDF2 with page-by-page timeout
    try:
        import PyPDF2
        logger.debug("Trying PyPDF2")
        
        with open(file_path, 'rb') as pdf_file:
            try:
                with time_limit(10):  # Just for creating the reader
                    reader = PyPDF2.PdfReader(pdf_file)
                    num_pages = len(reader.pages)
                    logger.debug(f"PDF has {num_pages} pages")
                    # Limit page count
                    max_pages = min(MAX_PAGE_COUNT, num_pages)
            except Exception as e:
                logger.warning(f"Error creating PDF reader: {e}")
                return ""
            
            # Extract text page by page with timeout for each page
            extracted_text = ""
            pages_processed = 0
            
            for page_num in range(max_pages):
                try:
                    with time_limit(3):  # 3 seconds per page maximum
                        page_text = reader.pages[page_num].extract_text() or ""
                        extracted_text += page_text + "\n"
                        pages_processed += 1
                        
                        # Check if we've reached size limit
                        if len(extracted_text) >= max_size:
                            logger.warning(f"Reached maximum text size at page {page_num+1}")
                            return extracted_text[:max_size]
                except TimeoutException:
                    logger.warning(f"Timeout extracting page {page_num+1}, skipping to next method")
                    break  # If one page times out, try next method
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num+1}: {e}")
                    continue  # Try next page
            
            if pages_processed > 0 and len(extracted_text.strip()) > 100:
                logger.debug(f"PyPDF2 extracted {pages_processed} pages")
                return extracted_text[:max_size]
    except ImportError:
        logger.debug("PyPDF2 not available")
    except Exception as e:
        logger.warning(f"PyPDF2 extraction failed: {e}")
    
    # Method 3: Try pdfplumber as a last resort
    try:
        import pdfplumber
        logger.debug("Trying pdfplumber")
        
        with time_limit(MAX_PDF_PROCESSING_TIME):
            with pdfplumber.open(file_path) as pdf:
                num_pages = len(pdf.pages)
                logger.debug(f"PDF has {num_pages} pages (pdfplumber)")
                # Limit page count
                max_pages = min(MAX_PAGE_COUNT, num_pages)
                
                # Extract text page by page with timeout for each page
                extracted_text = ""
                pages_processed = 0
                
                for i in range(max_pages):
                    try:
                        with time_limit(3):  # 3 seconds per page maximum
                            page_text = pdf.pages[i].extract_text() or ""
                            extracted_text += page_text + "\n"
                            pages_processed += 1
                            
                            # Check if we've reached size limit
                            if len(extracted_text) >= max_size:
                                logger.warning(f"Reached maximum text size at page {i+1}")
                                return extracted_text[:max_size]
                    except TimeoutException:
                        logger.warning(f"Timeout extracting page {i+1} with pdfplumber")
                        continue  # Try next page
                    except Exception as e:
                        logger.warning(f"Error extracting page {i+1} with pdfplumber: {e}")
                        continue  # Try next page
                
                if pages_processed > 0 and len(extracted_text.strip()) > 100:
                    logger.debug(f"pdfplumber extracted {pages_processed} pages")
                    return extracted_text[:max_size]
    except ImportError:
        logger.debug("pdfplumber not available")
    except TimeoutException:
        logger.warning("pdfplumber timed out")
    except Exception as e:
        logger.warning(f"pdfplumber extraction failed: {e}")
    
    # All methods failed
    logger.error(f"All PDF extraction methods failed for {file_path}")
    return ""


def extract_text_from_file(file_path, max_size=MAX_TEXT_SIZE):
    """
    Extract text from various file formats with aggressive timeouts.
    Drop-in replacement for the original extract_text_from_file function.
    
    Args:
        file_path: Path to the file
        max_size: Maximum text size to extract
        
    Returns:
        Extracted text or empty string if extraction fails
    """
    file_path = str(file_path)
    extension = os.path.splitext(file_path)[1].lower()
    
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
        return extract_text_from_pdf_safe(file_path, max_size)
    
    # Microsoft Office documents
    elif extension in ['.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']:
        try:
            # Try textract if available
            import textract
            with time_limit(60):  # 1 minute timeout for office docs
                text = textract.process(file_path).decode('utf-8', errors='ignore')
                return text[:max_size]
        except ImportError:
            logger.warning("textract not available for Office file extraction")
        except TimeoutException:
            logger.error(f"Timeout extracting text from Office file {file_path}")
            return ""
        except Exception as e:
            logger.error(f"Error extracting text from Office file {file_path}: {e}")
            return ""
        
        # Return empty string if textract failed
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


def test_pdf_extraction():
    """Test the PDF extraction on a directory of PDFs."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test PDF text extraction")
    parser.add_argument("--input", type=str, required=True, help="Input directory with PDFs")
    parser.add_argument("--output", type=str, default="pdf_extraction_results.txt", 
                        help="Output file for extraction results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        logger.error(f"Input directory not found: {args.input}")
        return 1
    
    # Find PDF files
    pdf_files = []
    for root, _, files in os.walk(args.input):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    # Test extraction on each file
    results = []
    success_count = 0
    
    for i, pdf_file in enumerate(pdf_files):
        logger.info(f"Processing file {i+1}/{len(pdf_files)}: {os.path.basename(pdf_file)}")
        
        start_time = time.time()
        text = extract_text_from_pdf_safe(pdf_file)
        elapsed_time = time.time() - start_time
        
        if text:
            success_count += 1
            status = "SUCCESS"
        else:
            status = "FAILED"
        
        results.append({
            "file": os.path.basename(pdf_file),
            "status": status,
            "time": elapsed_time,
            "text_length": len(text),
            "sample": text[:500].replace("\n", " ") if text else ""
        })
        
        logger.info(f"Status: {status}, Time: {elapsed_time:.2f}s, Length: {len(text)} chars")
    
    # Write results to file
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"PDF Extraction Results\n")
        f.write(f"====================\n")
        f.write(f"Total PDFs: {len(pdf_files)}\n")
        f.write(f"Successful extractions: {success_count}\n")
        f.write(f"Success rate: {success_count/len(pdf_files)*100:.1f}%\n\n")
        
        for result in results:
            f.write(f"File: {result['file']}\n")
            f.write(f"Status: {result['status']}\n")
            f.write(f"Time: {result['time']:.2f}s\n")
            f.write(f"Text length: {result['text_length']} chars\n")
            f.write(f"Sample: {result['sample']}\n")
            f.write("-" * 80 + "\n\n")
    
    logger.info(f"Results written to {args.output}")
    logger.info(f"Success rate: {success_count}/{len(pdf_files)} ({success_count/len(pdf_files)*100:.1f}%)")
    
    return 0


if __name__ == "__main__":
    sys.exit(test_pdf_extraction())