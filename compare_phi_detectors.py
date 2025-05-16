#!/usr/bin/env python
"""
Compare multiple PHI detection models on the same dataset.
"""
import os
import sys
import json
import warnings
import urllib3

# Disable SSL warnings and verification
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['PYTHONHTTPSVERIFY'] = '0'

# Try to handle SSL certificate issues
try:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

from src.genetic_algorithm import GeneticPIIDetector
from src.chromosome import DetectionGene, Chromosome

# Create regex baseline
def create_regex_baseline():
    genes = [
        DetectionGene(pattern=r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', pii_type="NAME"),
        DetectionGene(pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', pii_type="EMAIL"),
        DetectionGene(pattern=r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', pii_type="PHONE"),
        DetectionGene(pattern=r'\b\d{3}[-]?\d{2}[-]?\d{4}\b', pii_type="SSN")
    ]
    return Chromosome(genes=genes)

# Load text files from directory
def load_samples(directory):
    samples = []
    file_paths = []
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return samples, file_paths
    
    for file in os.listdir(directory):
        if file.endswith('.txt'):
            file_path = os.path.join(directory, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    if len(text) > 0:
                        samples.append(text)
                        file_paths.append(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    return samples, file_paths

# Try to load Presidio if available
presidio_available = False
try:
    from presidio_analyzer import AnalyzerEngine
    presidio_available = True
    presidio = AnalyzerEngine()
    print("Presidio loaded successfully")
except ImportError:
    print("Presidio not available. Install with: pip install presidio-analyzer")

# Try to load spaCy if available
spacy_available = False
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_lg")
        spacy_available = True
        print("spaCy loaded successfully with en_core_web_lg model")
    except:
        print("spaCy model not found, trying smaller model...")
        try:
            nlp = spacy.load("en_core_web_sm")
            spacy_available = True
            print("spaCy loaded successfully with en_core_web_sm model")
        except:
            print("Downloading spaCy model...")
            os.system("python -m spacy download en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
            spacy_available = True
except ImportError:
    print("spaCy not available. Install with: pip install spacy")

def main():
    """Run the model comparison."""
    # Parse arguments
    samples_dir = sys.argv[1] if len(sys.argv) > 1 else "medical_samples"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "model_comparison.json"
    generations = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    print(f"Loading samples from {samples_dir}...")
    samples, file_paths = load_samples(samples_dir)
    print(f"Loaded {len(samples)} samples")
    
    if not samples:
        print("No samples found. Exiting.")
        return 1
    
    # Train genetic algorithm
    print(f"Training genetic algorithm for {generations} generations...")
    genetic_model = GeneticPIIDetector(generations=generations)
    
    # Create empty annotations for training
    empty_annotations = [[] for _ in samples]
    genetic_model.train(samples, empty_annotations)
    
    # Create regex baseline
    regex_baseline = create_regex_baseline()
    
    # Run detection on each sample
    results = {
        "genetic": [],
        "regex": [],
        "presidio": [] if presidio_available else None,
        "spacy": [] if spacy_available else None
    }
    
    print("Running detection with all models...")
    for i, (sample, file_path) in enumerate(zip(samples, file_paths)):
        print(f"Processing sample {i+1}/{len(samples)}: {os.path.basename(file_path)}...")
        
        sample_result = {
            "file": os.path.basename(file_path),
            "file_path": file_path,
            "text_length": len(sample),
            "models": {}
        }
        
        # Genetic algorithm
        genetic_preds = genetic_model.predict(sample)
        results["genetic"].append({
            "file": os.path.basename(file_path),
            "count": len(genetic_preds),
            "types": list(set(p[3] for p in genetic_preds)),
            "entities": [{"text": p[0], "type": p[3], "confidence": p[4]} for p in genetic_preds]
        })
        
        # Regex baseline
        regex_preds = regex_baseline.detect(sample)
        results["regex"].append({
            "file": os.path.basename(file_path),
            "count": len(regex_preds),
            "types": list(set(p[3] for p in regex_preds)),
            "entities": [{"text": p[0], "type": p[3], "confidence": p[4]} for p in regex_preds]
        })
        
        # Presidio
        if presidio_available:
            try:
                presidio_results = presidio.analyze(text=sample, language="en")
                results["presidio"].append({
                    "file": os.path.basename(file_path),
                    "count": len(presidio_results),
                    "types": list(set(r.entity_type for r in presidio_results)),
                    "entities": [{"text": sample[r.start:r.end], "type": r.entity_type, "confidence": r.score} 
                                for r in presidio_results]
                })
            except Exception as e:
                print(f"Error with Presidio on {file_path}: {e}")
                results["presidio"].append({
                    "file": os.path.basename(file_path),
                    "count": 0,
                    "types": [],
                    "entities": [],
                    "error": str(e)
                })
        
        # spaCy NER
        if spacy_available:
            try:
                # Limit text length for spaCy to avoid memory issues
                if len(sample) > 100000:
                    sample_for_spacy = sample[:100000]
                    print(f"  Truncating text to 100K chars for spaCy processing")
                else:
                    sample_for_spacy = sample
                
                doc = nlp(sample_for_spacy)
                entities = [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
                results["spacy"].append({
                    "file": os.path.basename(file_path),
                    "count": len(entities),
                    "types": list(set(e[3] for e in entities)),
                    "entities": [{"text": e[0], "type": e[3], "confidence": 1.0} for e in entities]
                })
            except Exception as e:
                print(f"Error with spaCy on {file_path}: {e}")
                results["spacy"].append({
                    "file": os.path.basename(file_path),
                    "count": 0,
                    "types": [],
                    "entities": [],
                    "error": str(e)
                })
    
    # Calculate summary statistics
    summary = {}
    for model_name, model_results in results.items():
        if model_results is None:
            continue
            
        total_entities = sum(r["count"] for r in model_results)
        entity_types = set()
        for r in model_results:
            if "types" in r:  # Skip entries with errors
                entity_types.update(r["types"])
        
        summary[model_name] = {
            "total_entities": total_entities,
            "avg_per_sample": total_entities / len(samples) if samples else 0,
            "entity_types": list(entity_types)
        }
    
    # Calculate overlapping detections
    if len(summary) > 1:
        overlap_analysis = {}
        model_names = [m for m in results.keys() if results[m] is not None]
        
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                overlap_count = 0
                only_model1 = 0
                only_model2 = 0
                
                for sample_idx in range(len(samples)):
                    model1_entities = set(e["text"] for e in results[model1][sample_idx]["entities"])
                    model2_entities = set(e["text"] for e in results[model2][sample_idx]["entities"])
                    
                    overlap = model1_entities.intersection(model2_entities)
                    only_in_model1 = model1_entities - model2_entities
                    only_in_model2 = model2_entities - model1_entities
                    
                    overlap_count += len(overlap)
                    only_model1 += len(only_in_model1)
                    only_model2 += len(only_in_model2)
                
                overlap_analysis[f"{model1}_vs_{model2}"] = {
                    "overlap_count": overlap_count,
                    f"only_in_{model1}": only_model1,
                    f"only_in_{model2}": only_model2
                }
        
        summary["overlap_analysis"] = overlap_analysis
    
    # Write results to file
    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "num_samples": len(samples),
                "total_text_length": sum(len(s) for s in samples),
                "available_models": [m for m in results.keys() if results[m] is not None],
                "genetic_algorithm_generations": generations
            },
            "summary": summary,
            "detailed_results": results
        }, f, indent=2)
    
    print("\nCOMPARISON SUMMARY:")
    for model, stats in summary.items():
        if model == "overlap_analysis":
            continue
        print(f"\n{model.upper()}:")
        print(f"  Total entities: {stats['total_entities']}")
        print(f"  Avg per sample: {stats['avg_per_sample']:.2f}")
        print(f"  Entity types: {', '.join(stats['entity_types'])}")
    
    if "overlap_analysis" in summary:
        print("\nOVERLAP ANALYSIS:")
        for comparison, stats in summary["overlap_analysis"].items():
            print(f"  {comparison}:")
            for metric, value in stats.items():
                print(f"    {metric}: {value}")
    
    print(f"\nDetailed results saved to {output_file}")
    return 0

if __name__ == "__main__":
    sys.exit(main())