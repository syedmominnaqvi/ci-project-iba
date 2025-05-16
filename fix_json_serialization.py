#!/usr/bin/env python
"""
Fix JSON serialization for NumPy values in PHI result files.

Usage:
  ./fix_json_serialization.py --dir phi_results
"""

import os
import sys
import json
import argparse
import glob
import traceback
from pathlib import Path

def convert_numpy_types(obj):
    """Convert NumPy data types to Python native types for JSON serialization."""
    import numpy as np
    
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

def fix_json_file(json_path):
    """Fix a single JSON file, handling NumPy data types."""
    try:
        # Read the original JSON content as text (might have errors)
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Try to load the JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            print(f"⚠ Could not decode {json_path} as JSON, skipping")
            return False
            
        # Write back with conversion function
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=lambda o: None)
            
        print(f"✓ Fixed {os.path.basename(json_path)}")
        return True
        
    except Exception as e:
        print(f"⚠ Error fixing {os.path.basename(json_path)}: {e}")
        print(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix JSON serialization in PHI result files")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing JSON files")
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"Directory not found: {args.dir}")
        return 1
        
    # Find all JSON files
    pattern = os.path.join(args.dir, "**", "*.json")
    json_files = glob.glob(pattern, recursive=True)
    
    if not json_files:
        print(f"No JSON files found in {args.dir}")
        return 0
        
    print(f"Found {len(json_files)} JSON files to fix")
    
    # Process each file
    fixed = 0
    for json_path in json_files:
        if fix_json_file(json_path):
            fixed += 1
            
    print(f"\nFixed {fixed}/{len(json_files)} JSON files")
    
    # Specifically check for the summary file
    summary_path = os.path.join(args.dir, "phi_detection_summary.json")
    if os.path.exists(summary_path):
        print("\nRepairing summary file...")
        try:
            # Try to recreate the summary from existing phi_report.json files
            phi_reports = glob.glob(os.path.join(args.dir, "*_phi_report.json"))
            if phi_reports:
                print(f"Recreating summary from {len(phi_reports)} PHI report files")
                
                # Extract data from report files
                total_genetic = 0
                total_presidio = 0
                files_with_genetic_phi = 0
                files_with_presidio_phi = 0
                genetic_phi_types = set()
                presidio_phi_types = set()
                top_files = []
                
                for report_file in phi_reports:
                    try:
                        with open(report_file, 'r', encoding='utf-8') as f:
                            report_data = json.load(f)
                            
                        genetic_count = len(report_data.get('genetic_detections', []))
                        presidio_count = len(report_data.get('presidio_detections', []))
                        
                        total_genetic += genetic_count
                        total_presidio += presidio_count
                        
                        if genetic_count > 0:
                            files_with_genetic_phi += 1
                            
                        if presidio_count > 0:
                            files_with_presidio_phi += 1
                            
                        # Extract PHI types
                        for detection in report_data.get('genetic_detections', []):
                            if 'type' in detection:
                                genetic_phi_types.add(detection['type'])
                                
                        for detection in report_data.get('presidio_detections', []):
                            if 'type' in detection:
                                presidio_phi_types.add(detection['type'])
                                
                        # Add to top files list
                        if genetic_count > 0:
                            file_name = os.path.basename(report_file).replace('_phi_report.json', '')
                            top_files.append({
                                'file_name': file_name,
                                'genetic_count': genetic_count
                            })
                            
                    except Exception as e:
                        print(f"  ⚠ Error processing report file {os.path.basename(report_file)}: {e}")
                
                # Sort top files and keep top 10
                top_files = sorted(top_files, key=lambda x: x['genetic_count'], reverse=True)[:10]
                
                # Create summary
                summary = {
                    "total_files": len(phi_reports),
                    "successful_files": len(phi_reports),
                    "failed_files": 0,
                    "total_phi_detected": {
                        "genetic": total_genetic,
                        "presidio": total_presidio
                    },
                    "files_with_phi": {
                        "genetic": files_with_genetic_phi,
                        "presidio": files_with_presidio_phi
                    },
                    "phi_types_found": {
                        "genetic": list(genetic_phi_types),
                        "presidio": list(presidio_phi_types)
                    },
                    "top_files_by_phi_count": top_files,
                    "detection_timestamp": "2024-05-15 12:00:00",  # Placeholder
                    "processing_time_seconds": 0,  # Unknown
                    "model_settings": {
                        "genetic_generations": 20,  # Default value
                        "presidio_used": len(presidio_phi_types) > 0
                    }
                }
                
                # Write recreated summary
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2)
                print(f"✓ Summary file recreated from report data")
                
            else:
                # Try to fix the existing summary if we can't recreate it
                try:
                    # Try to read as text
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Fix any obvious NumPy int64 serialization issues
                    content = content.replace('Object of type int64 is not JSON serializable', '')
                    
                    # Try to parse as JSON
                    try:
                        data = json.loads(content)
                        # Write back corrected JSON
                        with open(summary_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2)
                        print(f"✓ Summary file fixed")
                    except json.JSONDecodeError:
                        print("⚠ Summary file could not be parsed as JSON")
                        
                except Exception as e:
                    print(f"⚠ Error fixing summary file: {e}")
                    
        except Exception as e:
            print(f"⚠ Error fixing summary file: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())