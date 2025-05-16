#!/usr/bin/env python
"""
Minimal PHI detector that uses a simplified genetic algorithm approach.
This version eliminates dependencies on external libraries for core functionality.
"""
import os
import sys
import json
import re
import random
import time
from datetime import datetime

class SimpleGene:
    """A simplified version of a detection gene."""
    
    def __init__(self, pattern, pii_type, confidence=0.7):
        self.pattern = pattern
        self.pii_type = pii_type
        self.confidence = confidence
        # Pre-compile pattern for better performance
        self.regex = re.compile(pattern)
    
    def detect(self, text):
        """Find matches in text."""
        matches = []
        for match in self.regex.finditer(text):
            start, end = match.span()
            matches.append((match.group(), start, end, self.pii_type, self.confidence))
        return matches
    
    def mutate(self):
        """Create a mutated copy of this gene."""
        # Simple mutation: just vary confidence a bit
        new_confidence = max(0.1, min(1.0, self.confidence + random.uniform(-0.1, 0.1)))
        return SimpleGene(self.pattern, self.pii_type, new_confidence)
    
    def __str__(self):
        return f"Gene({self.pii_type}, '{self.pattern}', conf={self.confidence:.2f})"


class SimpleChromosome:
    """A simplified version of a chromosome holding multiple genes."""
    
    def __init__(self, genes):
        self.genes = genes
    
    def detect(self, text):
        """Run detection on text using all genes."""
        all_matches = []
        for gene in self.genes:
            all_matches.extend(gene.detect(text))
        
        # Sort by start position
        all_matches.sort(key=lambda x: x[1])
        
        # Simple overlap resolution: keep non-overlapping matches
        final_matches = []
        last_end = -1
        for match in all_matches:
            if match[1] >= last_end:
                final_matches.append(match)
                last_end = match[2]
                
        return final_matches
    
    def mutate(self, mutation_rate=0.2):
        """Create a mutated copy of this chromosome."""
        if random.random() < mutation_rate:
            # Mutate some genes
            new_genes = [
                gene.mutate() if random.random() < 0.3 else gene 
                for gene in self.genes
            ]
            return SimpleChromosome(new_genes)
        else:
            return self
    
    def crossover(self, other):
        """Perform crossover with another chromosome."""
        if not self.genes or not other.genes:
            return self, other
        
        # Choose crossover point
        point1 = random.randint(0, len(self.genes))
        point2 = random.randint(0, len(other.genes))
        
        # Create offspring
        child1_genes = self.genes[:point1] + other.genes[point2:]
        child2_genes = other.genes[:point2] + self.genes[point1:]
        
        return SimpleChromosome(child1_genes), SimpleChromosome(child2_genes)


class SimpleGeneticDetector:
    """A simplified genetic algorithm for PHI detection."""
    
    def __init__(self, seed_patterns, population_size=20, generations=20):
        self.population_size = population_size
        self.generations = generations
        self.seed_patterns = seed_patterns
        self.best_chromosome = None
        
    def create_initial_population(self):
        """Create initial population with variations of the seed patterns."""
        population = []
        
        # First individual is the seed patterns exactly
        population.append(SimpleChromosome(self.seed_patterns))
        
        # Rest are variations
        for _ in range(self.population_size - 1):
            # Take a random subset of genes
            num_genes = random.randint(max(1, len(self.seed_patterns) // 2), len(self.seed_patterns))
            selected_genes = random.sample(self.seed_patterns, num_genes)
            
            # Create some mutations
            mutated_genes = [
                gene.mutate() if random.random() < 0.5 else gene 
                for gene in selected_genes
            ]
            
            population.append(SimpleChromosome(mutated_genes))
            
        return population
    
    def evaluate_fitness(self, chromosome, texts, annotations):
        """Evaluate chromosome fitness based on sample texts and annotations."""
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Evaluate on each text
        for text, annotations in zip(texts, annotations):
            # Get predictions
            predictions = chromosome.detect(text)
            
            # Convert predictions to a set of (start, end, type) tuples
            pred_spans = {(start, end, pii_type) for _, start, end, pii_type, _ in predictions}
            
            # Convert ground truth to a set of (start, end, type) tuples
            true_spans = {(start, end, pii_type) for start, end, pii_type in annotations}
            
            # Calculate metrics
            matches = pred_spans.intersection(true_spans)
            true_positives += len(matches)
            false_positives += len(pred_spans) - len(matches)
            false_negatives += len(true_spans) - len(matches)
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        
        # Simplistic fitness function: prefer precision slightly over recall
        fitness = (precision * 1.2 + recall) / 2.2 if precision + recall > 0 else 0
        
        return fitness, precision, recall
    
    def train(self, texts, annotations):
        """Train the genetic algorithm."""
        # Create initial population
        population = self.create_initial_population()
        
        # Track best chromosome
        best_fitness = -1
        self.best_chromosome = population[0]
        
        # Run for specified generations
        for generation in range(self.generations):
            print(f"Generation {generation + 1}/{self.generations}")
            
            # Evaluate fitness for each chromosome
            fitnesses = []
            for i, chromosome in enumerate(population):
                fitness, precision, recall = self.evaluate_fitness(chromosome, texts, annotations)
                fitnesses.append((fitness, chromosome, precision, recall))
                
                # Update best if better
                if fitness > best_fitness:
                    best_fitness = fitness
                    self.best_chromosome = chromosome
            
            # Print best in this generation
            fitnesses.sort(reverse=True)
            best_in_gen = fitnesses[0]
            print(f"  Best fitness: {best_in_gen[0]:.3f} (P={best_in_gen[2]:.3f}, R={best_in_gen[3]:.3f})")
            
            # Create next generation
            next_population = []
            
            # Elitism: keep the best individuals
            elite_count = max(1, self.population_size // 10)
            for i in range(elite_count):
                next_population.append(fitnesses[i][1])
            
            # Fill the rest through selection, crossover, and mutation
            while len(next_population) < self.population_size:
                # Tournament selection
                tournament_size = 3
                candidates = random.sample(fitnesses, tournament_size)
                candidates.sort(reverse=True)
                parent1 = candidates[0][1]
                
                candidates = random.sample(fitnesses, tournament_size)
                candidates.sort(reverse=True)
                parent2 = candidates[0][1]
                
                # Crossover
                if random.random() < 0.7:
                    child1, child2 = parent1.crossover(parent2)
                else:
                    child1, child2 = parent1, parent2
                
                # Mutation
                child1 = child1.mutate(mutation_rate=0.3)
                child2 = child2.mutate(mutation_rate=0.3)
                
                # Add to next generation
                next_population.append(child1)
                if len(next_population) < self.population_size:
                    next_population.append(child2)
            
            # Update population
            population = next_population
        
        # Final evaluation
        best_fitness, precision, recall = self.evaluate_fitness(
            self.best_chromosome, texts, annotations)
        print(f"\nTraining completed")
        print(f"Best chromosome: {len(self.best_chromosome.genes)} genes")
        print(f"Final fitness: {best_fitness:.3f} (P={precision:.3f}, R={recall:.3f})")
        
        return self.best_chromosome
    
    def predict(self, text):
        """Use the best chromosome to detect PHI."""
        if self.best_chromosome is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.best_chromosome.detect(text)


def create_base_patterns():
    """Create seed patterns for PHI detection."""
    patterns = [
        # Names
        SimpleGene(pattern=r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', pii_type="NAME"),
        SimpleGene(pattern=r'\bDr\.\s+[A-Z][a-z]+\b', pii_type="NAME"),
        
        # Emails
        SimpleGene(pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', pii_type="EMAIL"),
        
        # Phone numbers
        SimpleGene(pattern=r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', pii_type="PHONE"),
        SimpleGene(pattern=r'\(\d{3}\)\s*\d{3}[-.]?\d{4}\b', pii_type="PHONE"),
        
        # SSN
        SimpleGene(pattern=r'\b\d{3}[-]?\d{2}[-]?\d{4}\b', pii_type="SSN"),
        
        # Dates
        SimpleGene(pattern=r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', pii_type="DATE"),
        SimpleGene(pattern=r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b', pii_type="DATE"),
        
        # Addresses
        SimpleGene(pattern=r'\b\d+\s+[A-Z][a-z]+\s+(St|Ave|Rd|Blvd|Drive|Lane|Court|Way)\b', pii_type="ADDRESS"),
        
        # Medical records
        SimpleGene(pattern=r'\bMRN:?\s*\d+\b', pii_type="MEDICAL_RECORD"),
        SimpleGene(pattern=r'\bPatient ID:?\s*\d+\b', pii_type="MEDICAL_RECORD"),
    ]
    return patterns


def find_text_files(directory):
    """Find all text files in a directory."""
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return []
    
    text_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                text_files.append(os.path.join(root, file))
    
    return text_files


def read_file(file_path):
    """Read a text file safely."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def generate_synthetic_annotations(text, base_detector):
    """Generate annotations from base detector for training."""
    # Get detections from base patterns
    detections = base_detector.detect(text)
    
    # Convert to annotations format (start, end, type)
    annotations = [(start, end, pii_type) for _, start, end, pii_type, _ in detections]
    
    # Add some noise by removing some annotations
    if annotations and len(annotations) > 2:
        num_to_remove = max(1, len(annotations) // 5)
        indices_to_remove = random.sample(range(len(annotations)), num_to_remove)
        annotations = [ann for i, ann in enumerate(annotations) if i not in indices_to_remove]
    
    return annotations


def main():
    """Run minimal PHI detection."""
    # Parse arguments
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "minimal_results.json"
    max_files = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    try:
        # Find text files
        print(f"Finding text files in {input_dir}...")
        file_paths = find_text_files(input_dir)
        
        if not file_paths:
            print("No text files found!")
            return 1
        
        # Limit files if needed
        if max_files and len(file_paths) > max_files:
            print(f"Limiting to {max_files} files")
            random.shuffle(file_paths)
            file_paths = file_paths[:max_files]
            
        print(f"Found {len(file_paths)} text files")
        
        # Read files
        print("Reading files...")
        samples = []
        for file_path in file_paths:
            text = read_file(file_path)
            if text:
                samples.append((file_path, text))
        
        if not samples:
            print("No valid text content found!")
            return 1
        
        print(f"Successfully read {len(samples)} files")
        
        # Create base patterns
        print("Creating base patterns...")
        base_patterns = create_base_patterns()
        base_detector = SimpleChromosome(base_patterns)
        
        # Generate synthetic annotations for training
        print("Generating training annotations...")
        annotations = []
        for _, text in samples:
            annotations.append(generate_synthetic_annotations(text, base_detector))
        
        # Show annotation counts
        total_annotations = sum(len(a) for a in annotations)
        print(f"Generated {total_annotations} synthetic annotations " +
              f"(avg: {total_annotations/len(annotations):.1f} per file)")
        
        # Create genetic model
        print("Creating genetic detector...")
        genetic_detector = SimpleGeneticDetector(
            seed_patterns=base_patterns,
            population_size=20,
            generations=20
        )
        
        # Train model
        print("Training genetic detector...")
        start_time = time.time()
        genetic_detector.train([text for _, text in samples], annotations)
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.1f} seconds")
        
        # Process each sample
        print("Running detection on all samples...")
        results = {
            "genetic": [],
            "baseline": []
        }
        
        for file_path, text in samples:
            file_name = os.path.basename(file_path)
            print(f"Processing {file_name}...")
            
            # Genetic detector
            genetic_preds = genetic_detector.predict(text)
            
            # Base detector
            base_preds = base_detector.detect(text)
            
            # Record results
            results["genetic"].append({
                "file": file_name,
                "count": len(genetic_preds),
                "types": list(set(p[3] for p in genetic_preds)),
                "entities": [{"text": p[0], "type": p[3], "confidence": float(p[4])} for p in genetic_preds]
            })
            
            results["baseline"].append({
                "file": file_name,
                "count": len(base_preds),
                "types": list(set(p[3] for p in base_preds)),
                "entities": [{"text": p[0], "type": p[3], "confidence": float(p[4])} for p in base_preds]
            })
        
        # Calculate summary
        genetic_total = sum(r["count"] for r in results["genetic"])
        baseline_total = sum(r["count"] for r in results["baseline"])
        
        genetic_types = set()
        for r in results["genetic"]:
            genetic_types.update(r["types"])
            
        baseline_types = set()
        for r in results["baseline"]:
            baseline_types.update(r["types"])
        
        # Calculate overlap
        overlap_count = 0
        only_genetic = 0
        only_baseline = 0
        
        for i in range(len(samples)):
            genetic_entities = {e["text"] for e in results["genetic"][i]["entities"]}
            baseline_entities = {e["text"] for e in results["baseline"][i]["entities"]}
            
            overlap = genetic_entities.intersection(baseline_entities)
            overlap_count += len(overlap)
            only_genetic += len(genetic_entities - baseline_entities)
            only_baseline += len(baseline_entities - genetic_entities)
        
        # Save results
        output = {
            "metadata": {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "files_processed": len(samples),
                "training_time_seconds": training_time
            },
            "summary": {
                "genetic": {
                    "total_entities": genetic_total,
                    "avg_per_file": genetic_total / len(samples),
                    "entity_types": list(genetic_types)
                },
                "baseline": {
                    "total_entities": baseline_total,
                    "avg_per_file": baseline_total / len(samples),
                    "entity_types": list(baseline_types)
                },
                "overlap": {
                    "both_models": overlap_count,
                    "only_genetic": only_genetic,
                    "only_baseline": only_baseline
                }
            },
            "results": results
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)
        
        # Print summary
        print("\nRESULTS SUMMARY:")
        print(f"\nGenetic Algorithm:")
        print(f"  Total entities: {genetic_total}")
        print(f"  Avg per file: {genetic_total / len(samples):.2f}")
        print(f"  Entity types: {', '.join(genetic_types)}")
        
        print(f"\nBaseline:")
        print(f"  Total entities: {baseline_total}")
        print(f"  Avg per file: {baseline_total / len(samples):.2f}")
        print(f"  Entity types: {', '.join(baseline_types)}")
        
        print(f"\nOverlap Analysis:")
        print(f"  Detected by both models: {overlap_count}")
        print(f"  Only by genetic algorithm: {only_genetic}")
        print(f"  Only by baseline: {only_baseline}")
        
        print(f"\nResults saved to {output_file}")
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())