"""
Chromosome representation for PII/PHI detection.
"""
import random
import re
import numpy as np


class DetectionGene:
    """
    Represents a single gene in the chromosome, corresponding to a pattern
    that can detect a specific type of PII/PHI.
    """
    
    def __init__(self, pattern=None, context_window=5, pii_type=None, confidence=0.5):
        """
        Initialize a detection gene.
        
        Args:
            pattern: Regex pattern or None (will be randomly generated if None)
            context_window: Number of words to consider for context
            pii_type: Type of PII/PHI this gene detects
            confidence: Initial confidence score for this pattern
        """
        self.pii_types = [
            "NAME", "EMAIL", "PHONE", "SSN", "ADDRESS", 
            "MEDICAL_RECORD", "DIAGNOSIS", "MEDICATION"
        ]
        
        # Generate random pattern if not provided
        if pattern is None:
            self.pattern = self._generate_random_pattern()
        else:
            self.pattern = pattern
            
        self.context_window = max(1, context_window)
        self.pii_type = pii_type if pii_type else random.choice(self.pii_types)
        self.confidence = confidence
        
    def _generate_random_pattern(self):
        """Generate a random initial pattern based on common PII structures."""
        patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Name-like
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email-like
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone-like
            r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',  # SSN-like
            r'\b\d+\s+[A-Z][a-z]+\s+[A-Za-z]+\b',  # Address-like
            r'\b[A-Z]{2,}\b',  # Acronym
            r'\b\d{5,}\b',  # Number sequence
        ]
        return random.choice(patterns)
    
    def matches(self, text):
        """
        Check if this gene's pattern matches in the text.
        
        Args:
            text: String to check for matches
            
        Returns:
            List of tuples (match, start_pos, end_pos, confidence)
        """
        matches = []
        for match in re.finditer(self.pattern, text):
            start, end = match.span()
            matches.append((match.group(), start, end, self.confidence))
        return matches
    
    def mutate(self, mutation_rate=0.2):
        """
        Mutate this gene with some probability.
        
        Args:
            mutation_rate: Probability of mutation
            
        Returns:
            Mutated copy of this gene
        """
        if random.random() < mutation_rate:
            # Choose a mutation type
            mutation_type = random.choice([
                'pattern_expand', 'pattern_restrict', 
                'context_change', 'type_change', 'confidence_change'
            ])
            
            new_gene = DetectionGene(
                pattern=self.pattern,
                context_window=self.context_window,
                pii_type=self.pii_type,
                confidence=self.confidence
            )
            
            if mutation_type == 'pattern_expand':
                # Make pattern more general
                pattern_parts = self.pattern.split('\\b')
                if len(pattern_parts) > 1:
                    # Simplify one part of the pattern
                    i = random.randint(0, len(pattern_parts) - 1)
                    if pattern_parts[i] and pattern_parts[i] not in ['', '\\b']:
                        if random.random() < 0.5 and '[' in pattern_parts[i] and ']' in pattern_parts[i]:
                            # Expand character class
                            pattern_parts[i] = pattern_parts[i].replace('[A-Z]', '[A-Za-z]')
                        else:
                            # Make quantifier more flexible
                            pattern_parts[i] = pattern_parts[i].replace('{3}', '{2,4}')
                            pattern_parts[i] = pattern_parts[i].replace('+', '*')
                    new_gene.pattern = '\\b'.join(pattern_parts)
                    
            elif mutation_type == 'pattern_restrict':
                # Make pattern more specific
                if random.random() < 0.5:
                    # Add word boundary if not present
                    if not self.pattern.startswith('\\b'):
                        new_gene.pattern = '\\b' + self.pattern
                    if not self.pattern.endswith('\\b'):
                        new_gene.pattern = self.pattern + '\\b'
                else:
                    # Make quantifier more specific
                    new_gene.pattern = self.pattern.replace('*', '+')
                    new_gene.pattern = new_gene.pattern.replace('{2,}', '{2,4}')
                    
            elif mutation_type == 'context_change':
                # Change context window size
                new_gene.context_window = max(1, self.context_window + random.choice([-1, 1]))
                
            elif mutation_type == 'type_change':
                # Change PII type
                current_index = self.pii_types.index(self.pii_type) if self.pii_type in self.pii_types else 0
                new_index = (current_index + random.choice([-1, 1])) % len(self.pii_types)
                new_gene.pii_type = self.pii_types[new_index]
                
            elif mutation_type == 'confidence_change':
                # Adjust confidence
                confidence_delta = random.uniform(-0.1, 0.1)
                new_gene.confidence = max(0.1, min(1.0, self.confidence + confidence_delta))
                
            return new_gene
        else:
            return self
    
    def __str__(self):
        return f"DetectionGene(pattern='{self.pattern}', type='{self.pii_type}', confidence={self.confidence:.2f})"

    
class Chromosome:
    """
    Represents a complete solution for PII/PHI detection,
    consisting of multiple detection genes.
    """
    
    def __init__(self, genes=None, size=5):
        """
        Initialize a chromosome with detection genes.
        
        Args:
            genes: List of DetectionGene objects or None
            size: Number of genes to generate if genes is None
        """
        if genes is not None:
            self.genes = genes
        else:
            self.genes = [DetectionGene() for _ in range(size)]
        
    def detect(self, text):
        """
        Run detection using all genes in this chromosome.
        
        Args:
            text: String to detect PII/PHI in
            
        Returns:
            List of tuples (match, start_pos, end_pos, pii_type, confidence)
        """
        all_matches = []
        
        for gene in self.genes:
            matches = gene.matches(text)
            for match, start, end, confidence in matches:
                all_matches.append((match, start, end, gene.pii_type, confidence))
        
        # Sort by start position and resolve overlaps
        all_matches.sort(key=lambda x: x[1])
        
        # Resolve overlaps by keeping the higher confidence match
        final_matches = []
        i = 0
        while i < len(all_matches):
            current = all_matches[i]
            next_idx = i + 1
            
            # Check for overlap with next match
            while next_idx < len(all_matches) and all_matches[next_idx][1] < current[2]:
                next_match = all_matches[next_idx]
                
                # If overlapping, keep the one with higher confidence
                if next_match[1] < current[2]:  # Overlap exists
                    if next_match[4] > current[4]:  # Next has higher confidence
                        current = next_match
                
                next_idx += 1
            
            final_matches.append(current)
            i = next_idx
        
        return final_matches
    
    def mutate(self, mutation_rate=0.2, gene_mutation_rate=0.3):
        """
        Mutate this chromosome by mutating genes and potentially adding/removing genes.
        
        Args:
            mutation_rate: Overall mutation probability
            gene_mutation_rate: Probability of each gene mutating
            
        Returns:
            Mutated copy of this chromosome
        """
        if random.random() < mutation_rate:
            # Copy genes and mutate them
            new_genes = [gene.mutate(gene_mutation_rate) for gene in self.genes]
            
            # Occasionally add or remove a gene
            if random.random() < 0.2 and len(new_genes) > 1:
                # Remove a random gene
                del new_genes[random.randint(0, len(new_genes) - 1)]
            elif random.random() < 0.2:
                # Add a new gene
                new_genes.append(DetectionGene())
            
            return Chromosome(genes=new_genes)
        else:
            return self
    
    def crossover(self, other):
        """
        Perform crossover with another chromosome.
        
        Args:
            other: Another Chromosome object
            
        Returns:
            Two new Chromosome objects resulting from crossover
        """
        if not self.genes or not other.genes:
            return self, other
        
        # Choose crossover points
        point1 = random.randint(0, len(self.genes))
        point2 = random.randint(0, len(other.genes))
        
        # Create offspring
        child1_genes = self.genes[:point1] + other.genes[point2:]
        child2_genes = other.genes[:point2] + self.genes[point1:]
        
        return Chromosome(genes=child1_genes), Chromosome(genes=child2_genes)
    
    def __len__(self):
        return len(self.genes)
    
    def __str__(self):
        return f"Chromosome({len(self.genes)} genes)"