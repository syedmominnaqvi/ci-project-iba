"""
Main script for running the PII/PHI detection experiment.
"""
import os
import sys
import argparse
import json
import time
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from .data_utils import PIIDataset
from .genetic_algorithm import GeneticPIIDetector
from .baseline import PresidioBaseline, TransformerBaseline


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def run_experiment(args):
    """
    Run the full experiment comparing approaches.
    
    Args:
        args: Command line arguments
    """
    logger.info("Starting PII/PHI detection experiment")
    
    # Generate or load dataset
    dataset_handler = PIIDataset()

    if args.generate_data or not os.path.exists(os.path.join(args.data_dir, "train_texts.txt")):
        logger.info(f"Generating new dataset with {args.num_documents} documents")
        dataset = dataset_handler.generate_dataset(
            num_documents=args.num_documents,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio
        )
        # Save dataset with timestamp to keep history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(args.data_dir, f"dataset_{timestamp}")
        logger.info(f"Saving dataset to {save_dir}")
        dataset_handler.save_dataset(dataset, save_dir)

        # Also save to main data dir for default loading
        dataset_handler.save_dataset(dataset, args.data_dir)
    else:
        logger.info(f"Loading existing dataset from {args.data_dir}")
        dataset = dataset_handler.load_dataset(args.data_dir)
    
    # Set up models to compare
    models = {}
    
    # Add our evolutionary approach
    if args.run_genetic:
        logger.info("Setting up genetic algorithm model")
        genetic_model = GeneticPIIDetector(
            population_size=args.population_size,
            generations=args.generations,
            crossover_prob=args.crossover_prob,
            mutation_prob=args.mutation_prob,
            gene_mutation_prob=args.gene_mutation_prob,
            chromosome_size=args.chromosome_size
        )
        models["genetic"] = genetic_model
    
    # Add baselines
    if args.run_presidio:
        logger.info("Setting up Presidio baseline")
        try:
            presidio_model = PresidioBaseline()
            models["presidio"] = presidio_model
        except Exception as e:
            logger.error(f"Error setting up Presidio: {e}")
    
    if args.run_transformer:
        logger.info("Setting up Transformer baseline")
        try:
            transformer_model = TransformerBaseline(args.transformer_model)
            models["transformer"] = transformer_model
        except Exception as e:
            logger.error(f"Error setting up Transformer model: {e}")
    
    # Run experiments
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Running experiment with {model_name}")
        
        # Training (only for genetic model)
        if model_name == "genetic":
            logger.info("Training genetic model")
            start_time = time.time()
            model.train(
                dataset["train"]["texts"],
                dataset["train"]["annotations"]
            )
            train_time = time.time() - start_time
            logger.info(f"Training completed in {train_time:.2f} seconds")
        
        # Evaluation on test set
        logger.info(f"Evaluating {model_name} on test set")
        start_time = time.time()
        metrics = model.evaluate(
            dataset["test"]["texts"],
            dataset["test"]["annotations"]
        )
        eval_time = time.time() - start_time
        
        # Record results
        results[model_name] = {
            "metrics": metrics,
            "eval_time": eval_time
        }
        
        if model_name == "genetic":
            results[model_name]["train_time"] = train_time
        
        logger.info(f"{model_name} results: {metrics}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f"results_{timestamp}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Create comparison plots
    create_comparison_plots(results, os.path.join(args.output_dir, f"comparison_{timestamp}.png"))


def create_comparison_plots(results, output_file):
    """
    Create plots comparing models.
    
    Args:
        results: Dictionary of model results
        output_file: Path to save the plot
    """
    metrics = ["precision", "recall", "f1"]
    model_names = list(results.keys())
    
    # Extract metrics
    data = {metric: [] for metric in metrics}
    for model_name in model_names:
        for metric in metrics:
            data[metric].append(results[model_name]["metrics"][metric])
    
    # Create DataFrame
    df = pd.DataFrame(data, index=model_names)
    
    # Plot
    ax = df.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Model Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Metric")
    
    plt.tight_layout()
    plt.savefig(output_file)
    logger.info(f"Comparison plot saved to {output_file}")


def main():
    """Parse arguments and run experiment."""
    parser = argparse.ArgumentParser(description="PII/PHI Detection Experiment")
    
    # Dataset options
    parser.add_argument("--data_dir", type=str, default="data", help="Directory for data")
    parser.add_argument("--generate_data", action="store_true", help="Generate new dataset")
    parser.add_argument("--num_documents", type=int, default=100, help="Number of documents to generate")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Proportion for training")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Proportion for validation")
    
    # Model selection
    parser.add_argument("--run_genetic", action="store_true", help="Run genetic algorithm")
    parser.add_argument("--run_presidio", action="store_true", help="Run Presidio baseline")
    parser.add_argument("--run_transformer", action="store_true", help="Run Transformer baseline")
    
    # Genetic algorithm parameters
    parser.add_argument("--population_size", type=int, default=50, help="Population size")
    parser.add_argument("--generations", type=int, default=20, help="Number of generations")
    parser.add_argument("--crossover_prob", type=float, default=0.7, help="Crossover probability")
    parser.add_argument("--mutation_prob", type=float, default=0.2, help="Mutation probability")
    parser.add_argument("--gene_mutation_prob", type=float, default=0.3, help="Gene mutation probability")
    parser.add_argument("--chromosome_size", type=int, default=5, help="Initial chromosome size")
    
    # Transformer model options
    parser.add_argument("--transformer_model", type=str, default="dslim/bert-base-NER", help="Transformer model name")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Set defaults if no models selected
    if not (args.run_genetic or args.run_presidio or args.run_transformer):
        args.run_genetic = True
        args.run_presidio = True
        args.run_transformer = True
    
    run_experiment(args)


if __name__ == "__main__":
    main()