#!/usr/bin/env python
"""
Process MT Samples for PII/PHI detection.
This script:
1. Processes scraped MT Samples
2. Identifies potential PHI using rule-based methods
3. Creates annotations for training/testing
4. Prepares the dataset for use with our genetic algorithm
"""
import os
import re
import json
import argparse
import random
from datetime import datetime
import pandas as pd
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


def rule_based_annotation(text):
    """
    Use rules to annotate potential PHI in medical text.
    
    Args:
        text: String to annotate
        
    Returns:
        List of (start, end, pii_type) tuples
    """
    annotations = []
    
    # Common PHI patterns
    patterns = {
        # Names (Dr. X, Mr. Y, etc.)
        r'(?:Dr|Mr|Mrs|Ms|Miss|Doctor|Professor)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})': "NAME",
        
        # Dates (various formats)
        r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b': "DATE",
        r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b': "DATE",
        
        # Patient IDs and medical record numbers
        r'\b(?:patient|medical record|record|patient id|mrn|id)[:#\s]*([A-Z0-9]{5,})\b': "MEDICAL_RECORD",
        r'\b(?:pt|patient|medical record|record|id)[:#\s]*(\d{5,})\b': "MEDICAL_RECORD",
        
        # Phones
        r'\b(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b': "PHONE",
        
        # Ages
        r'\b(\d{1,3})[-\s](?:year|yr|y)[-\s]old\b': "AGE",
        
        # Addresses (simple patterns)
        r'\b(\d+\s+[A-Za-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Court|Ct|Lane|Ln|Way|Place|Pl))\b': "ADDRESS",
    }
    
    # Apply each pattern
    for pattern, phi_type in patterns.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            # Get the full match or the first group if it exists
            if match.groups():
                start, end = match.span(1)  # Use the first capture group
            else:
                start, end = match.span()   # Use the entire match
                
            # Add to annotations
            annotations.append((start, end, phi_type))
    
    return annotations


def use_presidio_for_annotation(text):
    """
    Use Microsoft Presidio for PHI detection.
    
    Args:
        text: String to analyze
        
    Returns:
        List of (start, end, pii_type) tuples
    """
    try:
        analyzer = AnalyzerEngine()
        
        # Analyze text with Presidio
        results = analyzer.analyze(
            text=text,
            entities=None,  # Detect all entity types
            language="en"
        )
        
        # Convert results to our format
        annotations = []
        
        # Mapping from Presidio entity types to our PII types
        type_mapping = {
            "PERSON": "NAME",
            "EMAIL_ADDRESS": "EMAIL",
            "PHONE_NUMBER": "PHONE",
            "US_SSN": "SSN",
            "LOCATION": "ADDRESS",
            "MEDICAL_LICENSE": "MEDICAL_RECORD",
            "DATE_TIME": "DATE",
            "US_PASSPORT": "ID",
            "US_DRIVER_LICENSE": "ID",
            "AGE": "AGE"
        }
        
        for result in results:
            entity_type = result.entity_type
            mapped_type = type_mapping.get(entity_type, entity_type)
            
            annotations.append((
                result.start,
                result.end,
                mapped_type
            ))
        
        return annotations
    
    except Exception as e:
        print(f"Error using Presidio: {e}")
        print("Falling back to rule-based annotation")
        return rule_based_annotation(text)


def process_mt_samples(input_dir, output_dir, use_presidio=True):
    """
    Process scraped MT samples and create annotated dataset.
    
    Args:
        input_dir: Directory with scraped samples
        output_dir: Directory to save processed data
        use_presidio: Whether to use Presidio or rule-based annotations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all text files
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.txt') and not f.endswith('_meta.txt')]
    
    if not all_files:
        print(f"No text files found in {input_dir}")
        return
    
    print(f"Found {len(all_files)} sample files")
    
    # Process each sample
    processed_samples = []
    
    for filename in all_files:
        filepath = os.path.join(input_dir, filename)
        
        # Read the sample
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Skip very short texts
        if len(text) < 100:
            print(f"Skipping {filename}: too short")
            continue
        
        print(f"Processing {filename} ({len(text)} characters)")
        
        # Get annotations
        if use_presidio:
            annotations = use_presidio_for_annotation(text)
        else:
            annotations = rule_based_annotation(text)
        
        print(f"  Found {len(annotations)} potential PHI entities")
        
        # Check for metadata
        meta_file = os.path.join(input_dir, filename.replace('.txt', '_meta.json'))
        metadata = {}
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except:
                print(f"  Could not read metadata file")
        
        # Save the processed sample
        output_file = os.path.join(output_dir, filename)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Save annotations
        ann_file = os.path.join(output_dir, filename.replace('.txt', '.ann'))
        with open(ann_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2)
        
        # Add to processed list
        processed_samples.append({
            "filename": filename,
            "text_length": len(text),
            "num_annotations": len(annotations),
            "annotation_types": list(set(a[2] for a in annotations)),
            "metadata": metadata
        })
    
    # Save index of all processed samples
    index_file = os.path.join(output_dir, "processed_index.json")
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(processed_samples, f, indent=2)
    
    # Create splits
    train_samples = []
    test_samples = []
    val_samples = []
    
    # Shuffle for random split
    random.shuffle(processed_samples)
    
    # 70% train, 15% validation, 15% test
    split1 = int(len(processed_samples) * 0.7)
    split2 = int(len(processed_samples) * 0.85)
    
    train_samples = processed_samples[:split1]
    val_samples = processed_samples[split1:split2]
    test_samples = processed_samples[split2:]
    
    # Save splits
    splits = {
        "train": [s["filename"] for s in train_samples],
        "val": [s["filename"] for s in val_samples],
        "test": [s["filename"] for s in test_samples]
    }
    
    splits_file = os.path.join(output_dir, "splits.json")
    with open(splits_file, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\nProcessing complete:")
    print(f"  Total samples: {len(processed_samples)}")
    print(f"  Training samples: {len(train_samples)}")
    print(f"  Validation samples: {len(val_samples)}")
    print(f"  Test samples: {len(test_samples)}")
    print(f"  Processed files saved to {output_dir}")


def main():
    """Parse arguments and process MT samples."""
    parser = argparse.ArgumentParser(description="Process MT Samples for PHI detection")
    
    parser.add_argument("--input_dir", type=str, default="mt_samples",
                      help="Directory with scraped samples")
    parser.add_argument("--output_dir", type=str, default="mt_processed",
                      help="Directory to save processed data")
    parser.add_argument("--use_presidio", action="store_true",
                      help="Use Presidio for annotations")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Input directory {args.input_dir} does not exist")
        return
    
    process_mt_samples(args.input_dir, args.output_dir, args.use_presidio)
    
    print("\nTo run the PHI detection on this dataset:")
    print(f"./load_custom_data.py --input {args.output_dir} --input_type directory")


if __name__ == "__main__":
    main()