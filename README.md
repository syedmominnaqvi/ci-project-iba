# Evolutionary Algorithm for PII/PHI Detection

An implementation of the research proposal "Evolutionary Algorithm Approach for Detecting Personal and Protected Health Information in Unstructured Data."

## Overview

This project uses genetic algorithms to evolve detection patterns for Personal Identifiable Information (PII) and Protected Health Information (PHI) in unstructured text data. Unlike traditional approaches that rely on predefined rules or supervised learning requiring extensive labeled data, this approach evolves detection patterns through natural selection principles.

## Key Features

- Flexible chromosome design for representing PII/PHI detection patterns
- Specialized genetic operators for evolving text patterns
- Comparison with industry standard baselines (Presidio and Transformer-based NER)
- Synthetic data generation for testing and evaluation

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd pii-detection-evolutionary

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Demo

Run the demo script to see the genetic algorithm in action with a small training set:

```bash
./demo.py
```

Options:
- `--train_size 50`: Change number of training documents
- `--generations 20`: Adjust evolution cycles
- `--custom_text "Patient John Smith has SSN 123-45-6789"`: Provide your own text to analyze

### Generate Sample Data

Generate and view synthetic PII/PHI documents:

```bash
./generate_samples.py --num_samples 10 --show_annotations
```

The samples will be saved to the `samples/` directory for further use.

### Visualize Detection

Visualize the detection results with colorized output:

```bash
./visualize_detection.py
```

Options:
- `--input_file samples/sample_1.txt`: Analyze a specific file
- `--model genetic`: Use only the genetic algorithm
- `--model presidio`: Use only the Presidio baseline
- `--model both`: Compare both approaches (default)

### Use Your Own Data

Use the custom data loader to train on your own dataset:

```bash
# For a directory of text files with separate annotation files
./load_custom_data.py --input your_data_dir/ --input_type directory

# For a CSV file with text and annotations
./load_custom_data.py --input your_data.csv --input_type csv --text_column "text" --ann_column "annotations"
```

### Full Experiment

Run the complete experiment with all models and evaluation:

```bash
./run_experiment.sh

# Or customize with specific parameters
python -m src.main \
  --data_dir data \
  --generate_data \
  --num_documents 200 \
  --run_genetic \
  --population_size 100 \
  --generations 50 \
  --output_dir results
```

## Project Structure

- `src/`: Source code
  - `chromosome.py`: Chromosome and gene representation
  - `genetic_algorithm.py`: Genetic algorithm implementation
  - `baseline.py`: Baseline models for comparison
  - `data_utils.py`: Data handling utilities
  - `main.py`: Main script for running experiments
- `tests/`: Test code
- `data/`: Data directory (created on first run)
- `results/`: Results directory (created on first run)
- `samples/`: Generated sample documents
- Utility scripts:
  - `demo.py`: Quick demonstration
  - `generate_samples.py`: Generate sample documents
  - `visualize_detection.py`: Visualize detection results
  - `load_custom_data.py`: Use custom datasets
  - `run_experiment.sh`: Run full experiment

## Running Tests

```bash
pytest tests/
```

## Key Parameters

- `--population_size`: Number of chromosomes in the population
- `--generations`: Number of generations to evolve
- `--crossover_prob`: Probability of crossover
- `--mutation_prob`: Probability of mutation
- `--gene_mutation_prob`: Probability of each gene mutating
- `--chromosome_size`: Initial number of genes in each chromosome

## Contribution

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.