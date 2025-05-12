"""
Baseline models for PII/PHI detection comparison.
"""
import re
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


class PresidioBaseline:
    """
    Microsoft Presidio-based baseline for PII detection.
    """
    
    def __init__(self):
        """Initialize the Presidio analyzer and anonymizer engines."""
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        
        # Mapping from Presidio entity types to our PII types
        self.type_mapping = {
            "PERSON": "NAME",
            "EMAIL_ADDRESS": "EMAIL",
            "PHONE_NUMBER": "PHONE",
            "US_SSN": "SSN",
            "LOCATION": "ADDRESS",
            "MEDICAL_LICENSE": "MEDICAL_RECORD",
            "HEALTH": "DIAGNOSIS",
            "MEDICAL_TREATMENT": "DIAGNOSIS",
            "US_ITIN": "SSN",
            "US_PASSPORT": "SSN",
            "US_BANK_NUMBER": "SSN",
            "US_DRIVER_LICENSE": "SSN",
            "CREDIT_CARD": "SSN",
            "DATE_TIME": "MEDICAL_RECORD",
            "NRP": "NAME",
            "URL": "EMAIL",
            "AGE": "MEDICAL_RECORD",
            "IBAN_CODE": "SSN",
            "ORGANIZATION": "NAME"
        }
    
    def detect(self, text):
        """
        Detect PII in text using Presidio.
        
        Args:
            text: String to analyze
            
        Returns:
            List of tuples (match, start_pos, end_pos, pii_type, confidence)
        """
        # Analyze text with Presidio
        results = self.analyzer.analyze(
            text=text,
            entities=None,  # Detect all entity types
            language="en"
        )
        
        # Convert results to our format
        detections = []
        for result in results:
            entity_type = result.entity_type
            mapped_type = self.type_mapping.get(entity_type, "OTHER")
            
            match_text = text[result.start:result.end]
            detections.append((
                match_text,
                result.start,
                result.end,
                mapped_type,
                result.score
            ))
        
        return detections
    
    def evaluate(self, texts, annotations):
        """
        Evaluate Presidio on test data.
        
        Args:
            texts: List of strings containing text
            annotations: List of lists of (start, end, pii_type) tuples
            
        Returns:
            Dictionary of metrics
        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Evaluate on each text
        for text, true_anns in zip(texts, annotations):
            predictions = self.detect(text)
            
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


class TransformerBaseline:
    """
    Transformer-based baseline for PII detection using a pre-trained NER model.
    """
    
    def __init__(self, model_name="dslim/bert-base-NER"):
        """
        Initialize the transformer model.
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")
        
        # Mapping from NER entity types to our PII types
        self.type_mapping = {
            "PER": "NAME",
            "PERSON": "NAME",
            "LOC": "ADDRESS",
            "LOCATION": "ADDRESS",
            "ORG": "NAME",
            "ORGANIZATION": "NAME",
            "MISC": "OTHER",
            "O": "OTHER",
            "B-PER": "NAME",
            "I-PER": "NAME",
            "B-LOC": "ADDRESS",
            "I-LOC": "ADDRESS",
            "B-ORG": "NAME",
            "I-ORG": "NAME",
            "B-MISC": "OTHER",
            "I-MISC": "OTHER",
        }
    
    def detect(self, text):
        """
        Detect PII in text using the transformer model.
        
        Args:
            text: String to analyze
            
        Returns:
            List of tuples (match, start_pos, end_pos, pii_type, confidence)
        """
        # Use the pipeline to detect entities
        results = self.nlp(text)
        
        # Convert results to our format
        detections = []
        for result in results:
            entity_type = result["entity_group"]
            mapped_type = self.type_mapping.get(entity_type, "OTHER")
            
            # Skip non-PII entities
            if mapped_type == "OTHER":
                continue
            
            detections.append((
                result["word"],
                result["start"],
                result["end"],
                mapped_type,
                result["score"]
            ))
        
        # Look for patterns not typically caught by NER models
        self._add_pattern_detections(text, detections)
        
        return detections
    
    def _add_pattern_detections(self, text, detections):
        """
        Add pattern-based detections for types that the NER model might miss.
        
        Args:
            text: Text to analyze
            detections: List to add detections to
        """
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            detections.append((
                match.group(),
                match.start(),
                match.end(),
                "EMAIL",
                0.95  # High confidence for regex pattern
            ))
        
        # Phone pattern
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        for match in re.finditer(phone_pattern, text):
            detections.append((
                match.group(),
                match.start(),
                match.end(),
                "PHONE",
                0.9
            ))
        
        # SSN pattern
        ssn_pattern = r'\b\d{3}[-]?\d{2}[-]?\d{4}\b'
        for match in re.finditer(ssn_pattern, text):
            detections.append((
                match.group(),
                match.start(),
                match.end(),
                "SSN",
                0.95
            ))
    
    def evaluate(self, texts, annotations):
        """
        Evaluate the transformer model on test data.
        
        Args:
            texts: List of strings containing text
            annotations: List of lists of (start, end, pii_type) tuples
            
        Returns:
            Dictionary of metrics
        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Evaluate on each text
        for text, true_anns in zip(texts, annotations):
            predictions = self.detect(text)
            
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