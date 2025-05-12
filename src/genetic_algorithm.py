"""
Genetic algorithm implementation for PII/PHI detection.
"""
import random
import numpy as np
from deap import base, creator, tools, algorithms
from .chromosome import Chromosome, DetectionGene


class GeneticPIIDetector:
    """
    Genetic algorithm-based approach to evolve PII/PHI detection patterns.
    """
    
    def __init__(self, 
                 population_size=50, 
                 generations=100,
                 crossover_prob=0.7,
                 mutation_prob=0.2,
                 gene_mutation_prob=0.3,
                 chromosome_size=5):
        """
        Initialize the genetic algorithm.
        
        Args:
            population_size: Number of chromosomes in the population
            generations: Number of generations to evolve
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation
            gene_mutation_prob: Probability of each gene mutating
            chromosome_size: Initial number of genes in each chromosome
        """
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.gene_mutation_prob = gene_mutation_prob
        self.chromosome_size = chromosome_size
        
        # DEAP setup
        creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, -0.5))  # Precision, Recall, Complexity
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("gene", DetectionGene)
        self.toolbox.register("chromosome", Chromosome, size=chromosome_size)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.chromosome, n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual, n=population_size)
        
        # Register genetic operators
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
        self.population = None
        self.best_individual = None
        self.training_texts = []
        self.training_annotations = []
    
    def _evaluate_individual(self, individual):
        """
        Evaluate fitness of an individual against training data.
        
        Args:
            individual: Individual to evaluate (contains one Chromosome)
            
        Returns:
            Tuple of (precision, recall, complexity)
        """
        if not self.training_texts or not self.training_annotations:
            return 0.0, 0.0, 1.0  # Default for no training data
        
        chromosome = individual[0]  # Extract the chromosome from the individual
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Total up metrics across all texts
        for text, annotations in zip(self.training_texts, self.training_annotations):
            # Get predictions from this chromosome
            predictions = chromosome.detect(text)
            
            # Convert predictions to a set of (start, end, type) tuples
            pred_spans = {(start, end, pii_type) for _, start, end, pii_type, _ in predictions}
            
            # Convert ground truth to a set of (start, end, type) tuples
            true_spans = {(start, end, pii_type) for start, end, pii_type in annotations}
            
            # Calculate true positives, false positives, false negatives
            matches = pred_spans.intersection(true_spans)
            true_positives += len(matches)
            false_positives += len(pred_spans) - len(matches)
            false_negatives += len(true_spans) - len(matches)
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        
        # Complexity penalty based on number of genes and pattern complexity
        num_genes = len(chromosome.genes)
        pattern_complexity = sum(len(gene.pattern) for gene in chromosome.genes) / max(1, num_genes)
        complexity = (num_genes * pattern_complexity) / 100  # Normalize
        
        return precision, recall, complexity
    
    def _crossover(self, ind1, ind2):
        """
        Perform crossover between two individuals.
        
        Args:
            ind1, ind2: Individuals to cross
            
        Returns:
            Modified individuals after crossover
        """
        if random.random() < self.crossover_prob:
            # Extract chromosomes
            chrom1 = ind1[0]
            chrom2 = ind2[0]
            
            # Perform crossover
            child1, child2 = chrom1.crossover(chrom2)
            
            # Update individuals
            ind1[0] = child1
            ind2[0] = child2
        
        return ind1, ind2
    
    def _mutate(self, individual):
        """
        Mutate an individual.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Modified individual after mutation
        """
        # Extract chromosome
        chromosome = individual[0]
        
        # Mutate chromosome
        individual[0] = chromosome.mutate(self.mutation_prob, self.gene_mutation_prob)
        
        return individual,
    
    def train(self, texts, annotations):
        """
        Train the genetic algorithm on annotated data.
        
        Args:
            texts: List of strings containing text
            annotations: List of lists of (start, end, pii_type) tuples
            
        Returns:
            Best individual found
        """
        self.training_texts = texts
        self.training_annotations = annotations
        
        # Initialize population
        self.population = self.toolbox.population()
        
        # Set up statistics to track
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        
        # Evolve population
        self.population, logbook = algorithms.eaSimple(
            self.population, 
            self.toolbox,
            cxpb=self.crossover_prob,
            mutpb=self.mutation_prob,
            ngen=self.generations,
            stats=stats,
            verbose=True
        )
        
        # Find best individual
        self.best_individual = tools.selBest(self.population, k=1)[0]
        
        return self.best_individual
    
    def predict(self, text):
        """
        Use the best chromosome to detect PII/PHI in text.
        
        Args:
            text: String to analyze
            
        Returns:
            List of detected PII/PHI items
        """
        if self.best_individual is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        best_chromosome = self.best_individual[0]
        return best_chromosome.detect(text)
    
    def evaluate(self, texts, annotations):
        """
        Evaluate model on test data.
        
        Args:
            texts: List of strings containing text
            annotations: List of lists of (start, end, pii_type) tuples
            
        Returns:
            Dictionary of metrics
        """
        if self.best_individual is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        best_chromosome = self.best_individual[0]
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Evaluate on each text
        for text, true_anns in zip(texts, annotations):
            predictions = best_chromosome.detect(text)
            
            # Convert to sets for comparison
            pred_spans = {(start, end, pii_type) for _, start, end, pii_type, _ in predictions}
            true_spans = {(start, end, pii_type) for start, end, pii_type in true_anns}
            
            # Calculate metrics
            matches = pred_spans.intersection(true_spans)
            true_positives += len(matches)
            false_positives += len(pred_spans) - len(matches)
            false_negatives += len(true_spans) - len(matches)
        
        # Calculate overall metrics
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }