# Claude Code Setup for PHI Detection Project

This document contains instructions and commands for Claude Code to effectively work with this project.

## Project Overview

This is an evolutionary algorithm for detecting Personal Health Information (PHI) and Personally Identifiable Information (PII) in unstructured data, implementing the research described in the included research proposal.

## Important Commands

### Run Tests
```bash
pytest tests/
```

### Run Demo
```bash
./demo.py
```

### Generate Medical Sample Data
```bash
./generate_medical_samples.py --num_documents 20 --output_dir medical_samples
```

### Run Simple PHI Detector (text files only)
```bash
./simple_phi_detector.py --input <directory> --output phi_results
```

### Run Full PHI Detector (with basic dependencies)
```bash
pip install PyPDF2 pandas tqdm
./process_any_data.py --input <directory> --output phi_results
```

### Model Comparison

### Simple Built-in Comparison
```bash
# The simple_phi_detector.py already compares genetic algorithm vs rule-based baseline
./simple_phi_detector.py --input medical_samples --output comparison_results
```

### Compare with Presidio (Free Microsoft PII detector)
```bash
# Install Presidio
pip install presidio-analyzer presidio-anonymizer

# Run comparison script
./process_any_data.py --input medical_samples --output presidio_comparison
```

### Custom Comparison Script
```bash
# Create a comparison script
cat > compare_phi_detectors.py << 'EOF'
#!/usr/bin/env python
"""
Compare multiple PHI detection models on the same dataset.
"""
import os
import sys
import json
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
    for file in os.listdir(directory):
        if file.endswith('.txt'):
            with open(os.path.join(directory, file), 'r') as f:
                samples.append(f.read())
    return samples

# Try to load Presidio if available
try:
    from presidio_analyzer import AnalyzerEngine
    presidio_available = True
    presidio = AnalyzerEngine()
except ImportError:
    presidio_available = False
    print("Presidio not available. Install with: pip install presidio-analyzer")

# Try to load spaCy if available
try:
    import spacy
    spacy_available = True
    try:
        nlp = spacy.load("en_core_web_lg")
    except:
        print("Downloading spaCy model...")
        os.system("python -m spacy download en_core_web_lg")
        nlp = spacy.load("en_core_web_lg")
except ImportError:
    spacy_available = False
    print("spaCy not available. Install with: pip install spacy")

# Load samples
samples_dir = sys.argv[1] if len(sys.argv) > 1 else "medical_samples"
print(f"Loading samples from {samples_dir}...")
samples = load_samples(samples_dir)
print(f"Loaded {len(samples)} samples")

# Train genetic algorithm
print("Training genetic algorithm...")
genetic_model = GeneticPIIDetector(generations=20)
genetic_model.train(samples, [[]])  # No annotations, just train on patterns

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
for i, sample in enumerate(samples):
    print(f"Processing sample {i+1}/{len(samples)}...")

    # Genetic algorithm
    genetic_preds = genetic_model.predict(sample)
    results["genetic"].append({
        "count": len(genetic_preds),
        "types": list(set(p[3] for p in genetic_preds)),
        "entities": [{"text": p[0], "type": p[3], "confidence": p[4]} for p in genetic_preds]
    })

    # Regex baseline
    regex_preds = regex_baseline.detect(sample)
    results["regex"].append({
        "count": len(regex_preds),
        "types": list(set(p[3] for p in regex_preds)),
        "entities": [{"text": p[0], "type": p[3], "confidence": p[4]} for p in regex_preds]
    })

    # Presidio
    if presidio_available:
        presidio_results = presidio.analyze(text=sample, language="en")
        results["presidio"].append({
            "count": len(presidio_results),
            "types": list(set(r.entity_type for r in presidio_results)),
            "entities": [{"text": sample[r.start:r.end], "type": r.entity_type, "confidence": r.score}
                         for r in presidio_results]
        })

    # spaCy NER
    if spacy_available:
        doc = nlp(sample)
        entities = [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        results["spacy"].append({
            "count": len(entities),
            "types": list(set(e[3] for e in entities)),
            "entities": [{"text": e[0], "type": e[3], "confidence": 1.0} for e in entities]
        })

# Calculate summary statistics
summary = {}
for model, model_results in results.items():
    if model_results is None:
        continue

    total_entities = sum(r["count"] for r in model_results)
    entity_types = set()
    for r in model_results:
        entity_types.update(r["types"])

    summary[model] = {
        "total_entities": total_entities,
        "avg_per_sample": total_entities / len(samples) if samples else 0,
        "entity_types": list(entity_types)
    }

# Write results to file
with open("model_comparison.json", "w") as f:
    json.dump({
        "summary": summary,
        "detailed_results": results
    }, f, indent=2)

print("\nCOMPARISON SUMMARY:")
for model, stats in summary.items():
    print(f"\n{model.upper()}:")
    print(f"  Total entities: {stats['total_entities']}")
    print(f"  Avg per sample: {stats['avg_per_sample']:.2f}")
    print(f"  Entity types: {', '.join(stats['entity_types'])}")

print("\nDetailed results saved to model_comparison.json")
EOF

# Make it executable
chmod +x compare_phi_detectors.py

# Run the comparison
./compare_phi_detectors.py medical_samples
```

### Interpreting Comparison Results

When comparing models, examine these metrics:

1. **Detection Count**: Total number of PHI entities found
   - Higher counts aren't necessarily better (could be false positives)

2. **PHI Types**: Types of entities detected by each model
   - Genetic algorithm often finds more diverse PHI types

3. **Unique Detections**: PHI found by one model but missed by others
   - Shows each model's strengths in pattern recognition

4. **Confidence Scores**: How certain each model is about detections
   - Higher confidence (>0.8) generally means more reliable

5. **False Positives**: Look for obvious errors in detections
   - Common words incorrectly flagged as PHI

A good evaluation compares both detection coverage (recall) and accuracy (precision). Without ground truth annotations, focus on examining the actual detected items and judge their correctness.

## Tips for Claude Code

1. When searching for files, use the Glob tool with patterns like `**/*.py` to find all Python files.

2. When working with the genetic algorithm, remember that fitness is measured by:
   - Precision (accuracy of matches)
   - Recall (coverage of all instances)
   - Complexity (simplicity of patterns)

3. Use the baseline comparison in reports to evaluate model performance.

4. Key project files:
   - `src/chromosome.py`: Core gene and chromosome structures
   - `src/genetic_algorithm.py`: Evolutionary algorithm implementation
   - `src/baseline.py`: Baseline models for comparison
   - `src/data_utils.py`: Data handling utilities

5. Use the pre-built test data in the `samples/` directory for quick testing.

## Model Training Commands

```python
from src.genetic_algorithm import GeneticPIIDetector

# Train model
model = GeneticPIIDetector(generations=30)
model.train(texts, annotations)

# Run detection
results = model.predict(text)

# Evaluate performance
metrics = model.evaluate(test_texts, test_annotations)
```

## Directory Structure

- `src/`: Source code
- `tests/`: Test code
- `samples/`: Generated sample documents
- `medical_samples/`: Generated medical records
- `phi_results/`: PHI detection results