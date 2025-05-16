#!/usr/bin/env python
"""
Patch script for process_any_data.py that fixes the PDF extraction issues.
"""
import os
import sys
import re
import shutil
from datetime import datetime

def main():
    """Patch the process_any_data.py script with fixed PDF extraction."""
    
    # Path to the original script
    original_script = "process_any_data.py"
    
    # Check if original script exists
    if not os.path.exists(original_script):
        print(f"Error: Original script {original_script} not found.")
        return 1
    
    # Create backup of original script
    backup_file = f"process_any_data.py.bak.{datetime.now().strftime('%Y%m%d%H%M%S')}"
    shutil.copy2(original_script, backup_file)
    print(f"Created backup: {backup_file}")
    
    # Load fix_pdf_extraction.py content
    pdf_fix_script = "fix_pdf_extraction.py"
    if not os.path.exists(pdf_fix_script):
        print(f"Error: PDF fix script {pdf_fix_script} not found.")
        return 1
    
    with open(pdf_fix_script, 'r') as f:
        pdf_fix_content = f.read()
    
    # Extract the extract_text_from_file function and extract_text_from_pdf_safe function
    extract_text_from_file_pattern = re.compile(
        r'def extract_text_from_file\(.*?\):.*?return ""', 
        re.DOTALL
    )
    extract_pdf_safe_pattern = re.compile(
        r'def extract_text_from_pdf_safe\(.*?\):.*?return ""', 
        re.DOTALL
    )
    timeout_class_pattern = re.compile(
        r'class TimeoutException\(Exception\):.*?pass', 
        re.DOTALL
    )
    time_limit_pattern = re.compile(
        r'@contextmanager\s+def time_limit\(seconds\):.*?signal\.alarm\(0\)', 
        re.DOTALL
    )
    
    # Extract functions from the fix script
    extract_text_from_file_match = extract_text_from_file_pattern.search(pdf_fix_content)
    extract_pdf_safe_match = extract_pdf_safe_pattern.search(pdf_fix_content)
    timeout_class_match = timeout_class_pattern.search(pdf_fix_content)
    time_limit_match = time_limit_pattern.search(pdf_fix_content)
    
    if not extract_text_from_file_match or not extract_pdf_safe_match:
        print("Error: Could not find required functions in the fix script.")
        return 1
    
    extract_text_from_file_code = extract_text_from_file_match.group(0)
    extract_pdf_safe_code = extract_pdf_safe_match.group(0)
    timeout_class_code = timeout_class_match.group(0) if timeout_class_match else ""
    time_limit_code = time_limit_match.group(0) if time_limit_match else ""
    
    # Load original script
    with open(original_script, 'r') as f:
        original_content = f.read()
    
    # Check if the original script already defines TimeoutException and time_limit
    has_timeout_exception = 'class TimeoutException' in original_content
    has_time_limit = '@contextmanager\ndef time_limit' in original_content
    
    # Find the extract_text_from_file function in the original script
    original_extract_func_pattern = re.compile(
        r'def extract_text_from_file\(.*?\):.*?return .*?"Format not supported for extraction"', 
        re.DOTALL
    )
    match = original_extract_func_pattern.search(original_content)
    
    if not match:
        print("Error: Could not find extract_text_from_file function in the original script.")
        return 1
    
    # Replace the function with our fixed version
    new_content = original_content
    
    # Add imports if needed
    if 'import signal' not in new_content:
        import_line = 'import os\nimport sys\nimport signal'
        new_content = new_content.replace('import os\nimport sys', import_line)
    
    if 'from contextlib import contextmanager' not in new_content:
        import_line = 'import tempfile\nimport subprocess\nfrom contextlib import contextmanager'
        new_content = new_content.replace('import tempfile\nimport subprocess', import_line)
    
    # Add TimeoutException class if needed
    if not has_timeout_exception:
        # Find the right place to add the class - after imports, before first class or function
        first_def = new_content.find('def ')
        first_class = new_content.find('class ')
        insert_pos = min(x for x in [first_def, first_class] if x >= 0)
        
        # Add constants and the TimeoutException class
        constants = """
# Constants for safe PDF processing
MAX_TEXT_SIZE = 1 * 1024 * 1024  # 1MB max text size
MAX_PDF_PROCESSING_TIME = 30  # 30 seconds timeout for PDF processing
MAX_PAGE_COUNT = 50  # Maximum pages to process in a PDF

"""
        new_content = new_content[:insert_pos] + constants + timeout_class_code + "\n\n" + new_content[insert_pos:]
    
    # Add time_limit function if needed
    if not has_time_limit:
        # Find where to add the time_limit function - after TimeoutException class
        if not has_timeout_exception:
            # We just added it, so find it now
            timeout_pos = new_content.find('class TimeoutException')
            class_end = new_content.find('\n\n', timeout_pos)
            insert_pos = class_end + 2
        else:
            # It already exists, find it and add after it
            timeout_pos = new_content.find('class TimeoutException')
            class_end = new_content.find('\n\n', timeout_pos)
            insert_pos = class_end + 2
        
        # Add the time_limit function
        new_content = new_content[:insert_pos] + time_limit_code + "\n\n" + new_content[insert_pos:]
    
    # Replace the extract_text_from_file function
    new_content = original_extract_func_pattern.sub(extract_text_from_file_code, new_content)
    
    # Add the extract_text_from_pdf_safe function after extract_text_from_file
    extract_text_pos = new_content.find(extract_text_from_file_code) + len(extract_text_from_file_code)
    new_content = new_content[:extract_text_pos] + "\n\n\n" + extract_pdf_safe_code + new_content[extract_text_pos:]
    
    # Add logging setup if it doesn't exist
    if 'logging.basicConfig' not in new_content:
        import_logging = 'import logging\n'
        if 'import tempfile' in new_content:
            new_content = new_content.replace('import tempfile', import_logging + 'import tempfile')
        else:
            # Add after other imports
            import_section_end = new_content.find('\n\n', new_content.find('import'))
            new_content = new_content[:import_section_end] + '\n' + import_logging + new_content[import_section_end:]
        
        # Add logging setup
        logging_setup = """
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
"""
        # Find where to add logging setup - after imports
        import_section_end = new_content.find('\n\n', new_content.find('import'))
        new_content = new_content[:import_section_end] + logging_setup + new_content[import_section_end:]
    
    # Modify the call to process_files to use a smaller batch size
    process_call_pattern = re.compile(r'process_files\(\s*file_paths,\s*args\.output,\s*generations=args\.generations,\s*use_presidio=not args\.skip_presidio,\s*batch_size=args\.batch_size\s*\)')
    if process_call_pattern.search(new_content):
        new_process_call = 'process_files(\n        file_paths,\n        args.output,\n        generations=args.generations,\n        use_presidio=not args.skip_presidio,\n        batch_size=5  # Reduced batch size for stability\n    )'
        new_content = process_call_pattern.sub(new_process_call, new_content)
    
    # Write the updated script
    with open(original_script, 'w') as f:
        f.write(new_content)
    
    print(f"Successfully patched {original_script} with fixed PDF extraction.")
    print("The script now has:")
    print("  - Proper timeout handling for PDF processing")
    print("  - Page-by-page extraction with individual timeouts")
    print("  - Maximum page limit to prevent hanging")
    print("  - Reduced batch size for better stability")
    print("\nYou can now run the script again with:")
    print(f"  ./process_any_data.py --input your_directory --output phi_results")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())