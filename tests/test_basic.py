"""
Basic tests for the PII/PHI detection modules.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.chromosome import DetectionGene, Chromosome
from src.data_utils import PIIDataset


def test_detection_gene():
    """Test basic functionality of a detection gene."""
    # Create a gene with a specific pattern
    gene = DetectionGene(pattern=r'\b[A-Z][a-z]+\b', pii_type="NAME")
    
    # Test matching
    text = "Hello John Smith, how are you today?"
    matches = gene.matches(text)
    
    # Should match "Hello", "John", and "Smith"
    assert len(matches) == 3
    assert matches[0][0] == "Hello"
    assert matches[1][0] == "John"
    assert matches[2][0] == "Smith"
    assert matches[0][3] == gene.confidence  # Check confidence is passed

    # Test non-matching
    text = "hello world"
    matches = gene.matches(text)
    assert len(matches) == 0
    
    # Test mutation
    mutated = gene.mutate(mutation_rate=1.0)  # Force mutation
    assert mutated.pattern != gene.pattern or mutated.context_window != gene.context_window or mutated.confidence != gene.confidence


def test_chromosome():
    """Test the chromosome functionality."""
    # Create a chromosome with specific genes
    gene1 = DetectionGene(pattern=r'\b[A-Z][a-z]+\b', pii_type="NAME")
    gene2 = DetectionGene(pattern=r'\b\d{3}-\d{2}-\d{4}\b', pii_type="SSN")
    chromosome = Chromosome(genes=[gene1, gene2])
    
    # Test detection
    text = "Patient John with SSN 123-45-6789 visited the clinic."
    matches = chromosome.detect(text)
    
    # Should have matches for "Patient", "John", and the SSN
    assert len(matches) == 3
    
    # Verify name matches
    name_matches = [m for m in matches if m[3] == "NAME"]
    assert len(name_matches) == 2
    assert "Patient" in [m[0] for m in name_matches]
    assert "John" in [m[0] for m in name_matches]
    
    # Verify SSN match
    ssn_match = next((m for m in matches if m[3] == "SSN"), None)
    assert ssn_match is not None
    assert ssn_match[0] == "123-45-6789"
    
    # Test mutation
    mutated = chromosome.mutate(mutation_rate=1.0, gene_mutation_rate=1.0)  # Force mutation
    assert len(mutated.genes) != 0
    
    # Test crossover
    other_gene = DetectionGene(pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', pii_type="EMAIL")
    other_chromosome = Chromosome(genes=[other_gene])
    
    child1, child2 = chromosome.crossover(other_chromosome)
    assert isinstance(child1, Chromosome)
    assert isinstance(child2, Chromosome)


def test_data_generation():
    """Test the data generation functionality."""
    dataset = PIIDataset()
    
    # Generate a single document
    doc, annotations = dataset.generate_synthetic_document()
    
    assert isinstance(doc, str)
    assert len(doc) > 0
    assert isinstance(annotations, list)
    assert len(annotations) > 0
    
    # Verify annotations format
    for start, end, pii_type in annotations:
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert isinstance(pii_type, str)
        
        # Check that annotation points to correct text
        span_text = doc[start:end]
        assert len(span_text) > 0
    
    # Test dataset generation
    full_dataset = dataset.generate_dataset(num_documents=5)
    
    assert "train" in full_dataset
    assert "val" in full_dataset
    assert "test" in full_dataset
    
    assert len(full_dataset["train"]["texts"]) > 0
    assert len(full_dataset["train"]["annotations"]) == len(full_dataset["train"]["texts"])


if __name__ == "__main__":
    test_detection_gene()
    test_chromosome()
    test_data_generation()
    print("All tests passed!")