#!/usr/bin/env python
"""
Generate and display sample synthetic documents for PII/PHI detection.
"""
import os
import argparse
import json
from src.data_utils import PIIDataset

def main():
    """Generate and display sample data."""
    parser = argparse.ArgumentParser(description="Generate sample PII/PHI documents")
    parser.add_argument("--num_samples", type=int, default=5, 
                       help="Number of sample documents to generate")
    parser.add_argument("--save_dir", type=str, default="samples",
                       help="Directory to save samples")
    parser.add_argument("--show_annotations", action="store_true",
                       help="Show annotations with documents")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Generate samples
    print(f"Generating {args.num_samples} sample documents...\n")
    
    dataset = PIIDataset()
    samples = []
    
    for i in range(args.num_samples):
        doc, annotations = dataset.generate_synthetic_document()
        samples.append({"text": doc, "annotations": annotations})
        
        # Save to individual files
        with open(os.path.join(args.save_dir, f"sample_{i+1}.txt"), "w") as f:
            f.write(doc)
        
        with open(os.path.join(args.save_dir, f"sample_{i+1}_annotations.json"), "w") as f:
            json.dump(annotations, f, indent=2)
        
        # Display the sample
        print(f"=== SAMPLE {i+1} ===")
        print(doc)
        print()
        
        if args.show_annotations:
            print("Annotations:")
            for start, end, pii_type in annotations:
                print(f"- {pii_type}: '{doc[start:end]}' (positions {start}-{end})")
            print()
        
        print("-" * 80)
        print()
    
    # Save all samples to one file for convenience
    with open(os.path.join(args.save_dir, "all_samples.json"), "w") as f:
        json.dump(samples, f, indent=2)
    
    print(f"All samples saved to {args.save_dir}/")

if __name__ == "__main__":
    main()