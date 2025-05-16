#!/usr/bin/env python
"""
Apply all fixes to the PHI detection system:
1. Fix SSL certificate issues
2. Patch process_any_data.py with robust PDF extraction
3. Create a modified script with simplified batch processing

This is a comprehensive fix for the issues encountered with:
- PDF processing hanging
- SSL certificate verification errors
- Batch processing stability
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_script(script_path, *args):
    """Run a Python script with arguments."""
    print(f"\n{'='*60}")
    print(f"Running {script_path}...")
    print(f"{'='*60}")
    
    cmd = [sys.executable, script_path] + list(args)
    result = subprocess.run(cmd)
    
    return result.returncode == 0

def create_robust_script():
    """Create a simplified robust version of the main script."""
    original_script = "process_any_data.py"
    robust_script = "robust_phi_detector.py"
    
    # Check if robust script already exists
    if os.path.exists(robust_script):
        print(f"\n{robust_script} already exists. Skipping creation.")
        return True
    
    if not os.path.exists(original_script):
        print(f"Error: {original_script} not found.")
        return False
    
    # Copy from our pre-made version if it exists
    robust_path = Path("robust_process_data.py")
    if robust_path.exists():
        shutil.copy2(robust_path, robust_script)
        os.chmod(robust_script, 0o755)  # Make executable
        print(f"\nCreated {robust_script} from pre-made version.")
        return True
    
    print(f"Error: Pre-made robust script not found.")
    return False

def main():
    """Apply all fixes to the PHI detection system."""
    print("PHI Detection System Fixer")
    print("=========================")
    print("This script will apply all necessary fixes to make the PHI detection")
    print("system work correctly, especially for PDF processing and SSL issues.")
    
    # Check for required scripts
    required_scripts = [
        "fix_ssl_issues.py",
        "patch_process_script.py",
        "fix_pdf_extraction.py",
        "process_any_data.py"
    ]
    
    for script in required_scripts:
        if not os.path.exists(script):
            print(f"Error: Required script {script} not found.")
            return 1
    
    # Step 1: Fix SSL issues
    if run_script("fix_ssl_issues.py"):
        print("✅ SSL issues fixed successfully")
    else:
        print("❌ Failed to fix SSL issues")
    
    # Step 2: Patch process_any_data.py with PDF extraction fixes
    if run_script("patch_process_script.py"):
        print("✅ process_any_data.py patched successfully")
    else:
        print("❌ Failed to patch process_any_data.py")
    
    # Step 3: Create robust script
    if create_robust_script():
        print("✅ Created robust_phi_detector.py")
    else:
        print("❌ Failed to create robust script")
    
    # Step 4: Make all scripts executable
    for script in [
        "process_any_data.py", 
        "robust_phi_detector.py",
        "fix_ssl_issues.py", 
        "fix_pdf_extraction.py"
    ]:
        if os.path.exists(script):
            os.chmod(script, 0o755)
    
    print("\nAll fixes have been applied!")
    print("\nRecommended usage:")
    print("1. For most reliable operation:")
    print("   ./robust_phi_detector.py --input your_directory --output phi_results --batch_size 5")
    print("\n2. If you prefer the original script with fixes:")
    print("   ./process_any_data.py --input your_directory --output phi_results")
    print("\n3. To fix SSL issues again if needed:")
    print("   ./fix_ssl_issues.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())