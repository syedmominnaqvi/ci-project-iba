"""
Utilities for handling training and test data.
"""
import os
import json
import random
import pandas as pd


class PIIDataset:
    """
    Class for loading, generating, and managing PII/PHI datasets.
    """
    
    def __init__(self):
        """Initialize the dataset handler."""
        # Sample data for generating synthetic PII
        self.first_names = [
            "John", "Jane", "Michael", "Sarah", "David", "Emily", "Robert", "Lisa",
            "James", "Linda", "William", "Patricia", "Richard", "Jennifer", "Joseph",
            "Elizabeth", "Thomas", "Barbara", "Charles", "Margaret"
        ]
        
        self.last_names = [
            "Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson",
            "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin",
            "Thompson", "Garcia", "Martinez", "Robinson"
        ]
        
        self.streets = [
            "Main St", "Oak Ave", "Maple Dr", "Cedar Ln", "Pine St", "Elm St", "Washington Ave",
            "Park Rd", "Lake Dr", "River Rd", "Church St", "Highland Ave", "Sunset Blvd",
            "Broadway", "1st St", "2nd Ave", "3rd St", "4th Ave", "Center St", "Market St"
        ]
        
        self.cities = [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", 
            "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
            "Fort Worth", "Columbus", "San Francisco", "Charlotte", "Indianapolis", 
            "Seattle", "Denver", "Boston"
        ]
        
        self.states = [
            "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
            "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD"
        ]
        
        self.diagnoses = [
            "Hypertension", "Type 2 Diabetes", "Asthma", "COPD", "Arthritis",
            "Depression", "Anxiety Disorder", "Hypothyroidism", "Hyperlipidemia",
            "Gastroesophageal Reflux Disease", "Migraine", "Osteoporosis",
            "Chronic Kidney Disease", "Sleep Apnea", "Eczema", "Psoriasis",
            "Chronic Fatigue Syndrome", "Irritable Bowel Syndrome", "Fibromyalgia",
            "Allergic Rhinitis"
        ]
        
        self.medications = [
            "Lisinopril", "Metformin", "Amlodipine", "Atorvastatin", "Levothyroxine",
            "Albuterol", "Omeprazole", "Metoprolol", "Losartan", "Gabapentin",
            "Hydrochlorothiazide", "Sertraline", "Simvastatin", "Montelukast",
            "Fluticasone", "Pantoprazole", "Escitalopram", "Prednisone", "Furosemide",
            "Rosuvastatin"
        ]
    
    def _generate_name(self):
        """Generate a random full name."""
        first = random.choice(self.first_names)
        last = random.choice(self.last_names)
        return f"{first} {last}"
    
    def _generate_email(self, name=None):
        """Generate a random email address."""
        if name is None:
            name = self._generate_name()
        
        first, last = name.split()
        domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "example.com"]
        
        email_type = random.choice([
            f"{first.lower()}.{last.lower()}@{random.choice(domains)}",
            f"{first.lower()}{last.lower()}@{random.choice(domains)}",
            f"{first.lower()}{random.randint(1, 99)}@{random.choice(domains)}",
            f"{last.lower()}.{first.lower()[0]}@{random.choice(domains)}"
        ])
        
        return email_type
    
    def _generate_phone(self):
        """Generate a random phone number."""
        formats = [
            f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}",
            f"{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
            f"{random.randint(100, 999)}.{random.randint(100, 999)}.{random.randint(1000, 9999)}"
        ]
        return random.choice(formats)
    
    def _generate_ssn(self):
        """Generate a random SSN."""
        formats = [
            f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}",
            f"{random.randint(100, 999)} {random.randint(10, 99)} {random.randint(1000, 9999)}"
        ]
        return random.choice(formats)
    
    def _generate_address(self):
        """Generate a random address."""
        formats = [
            f"{random.randint(100, 9999)} {random.choice(self.streets)}, {random.choice(self.cities)}, {random.choice(self.states)} {random.randint(10000, 99999)}",
            f"{random.randint(100, 9999)} {random.choice(self.streets)}\n{random.choice(self.cities)}, {random.choice(self.states)} {random.randint(10000, 99999)}"
        ]
        return random.choice(formats)
    
    def _generate_medical_record(self):
        """Generate a random medical record number."""
        formats = [
            f"MRN-{random.randint(100000, 999999)}",
            f"#{random.randint(1000000, 9999999)}",
            f"PATIENT-{random.randint(10000, 99999)}"
        ]
        return random.choice(formats)
    
    def generate_synthetic_document(self, include_pii=True):
        """
        Generate a synthetic document with PII/PHI.
        
        Args:
            include_pii: Whether to include PII in the document
            
        Returns:
            Tuple of (document text, list of annotations)
        """
        templates = [
            "Patient Information:\nName: {name}\nDate of Birth: {dob}\nPhone: {phone}\nEmail: {email}\nAddress: {address}\nSSN: {ssn}\nMedical Record Number: {mrn}\n\nDiagnosis: {diagnosis}\nMedications: {medications}",
            
            "MEDICAL REPORT\n\nPatient: {name}\nMRN: {mrn}\nDOB: {dob}\nContact: {phone}, {email}\nHome: {address}\nSocial Security: {ssn}\n\nClinical Summary:\nThe patient was diagnosed with {diagnosis} and prescribed {medications}.",
            
            "PATIENT CASE NOTES\n\n{name} ({dob}) presented with symptoms consistent with {diagnosis}. Current treatment includes {medications}.\n\nContact information: {phone}, {email}\nResidence: {address}\nIdentification: SSN {ssn}, Medical Record {mrn}",
            
            "Dear {name},\n\nThank you for your recent visit on {date}. This is a summary of your appointment.\n\nYour diagnosis is {diagnosis}. I've prescribed {medications} for your condition.\n\nPlease contact us at 555-123-4567 if you have any questions.\n\nSincerely,\nDr. {doctor_name}\n\nPatient Details:\nPhone: {phone}\nEmail: {email}\nAddress: {address}\nSS#: {ssn}\nMRN: {mrn}"
        ]
        
        # Generate PII data
        name = self._generate_name()
        dob = f"{random.randint(1, 12)}/{random.randint(1, 28)}/{random.randint(1940, 2010)}"
        phone = self._generate_phone()
        email = self._generate_email(name)
        address = self._generate_address()
        ssn = self._generate_ssn()
        mrn = self._generate_medical_record()
        diagnosis = random.choice(self.diagnoses)
        medications = random.choice(self.medications)
        date = f"{random.randint(1, 12)}/{random.randint(1, 28)}/{random.randint(2022, 2023)}"
        doctor_name = self._generate_name()
        
        # Create document
        template = random.choice(templates)
        document = template.format(
            name=name,
            dob=dob,
            phone=phone,
            email=email,
            address=address,
            ssn=ssn,
            mrn=mrn,
            diagnosis=diagnosis,
            medications=medications,
            date=date,
            doctor_name=doctor_name
        )
        
        # If we don't want PII, redact it
        if not include_pii:
            document = document.replace(name, "[REDACTED-NAME]")
            document = document.replace(phone, "[REDACTED-PHONE]")
            document = document.replace(email, "[REDACTED-EMAIL]")
            document = document.replace(address, "[REDACTED-ADDRESS]")
            document = document.replace(ssn, "[REDACTED-SSN]")
            document = document.replace(mrn, "[REDACTED-MRN]")
            return document, []
        
        # Create annotations
        annotations = []
        
        # Find name in document
        start = document.find(name)
        if start != -1:
            annotations.append((start, start + len(name), "NAME"))
        
        # Find phone in document
        start = document.find(phone)
        if start != -1:
            annotations.append((start, start + len(phone), "PHONE"))
        
        # Find email in document
        start = document.find(email)
        if start != -1:
            annotations.append((start, start + len(email), "EMAIL"))
        
        # Find address in document (might span multiple lines)
        start = document.find(address)
        if start != -1:
            annotations.append((start, start + len(address), "ADDRESS"))
        
        # Find SSN in document
        start = document.find(ssn)
        if start != -1:
            annotations.append((start, start + len(ssn), "SSN"))
        
        # Find MRN in document
        start = document.find(mrn)
        if start != -1:
            annotations.append((start, start + len(mrn), "MEDICAL_RECORD"))
        
        # Find diagnosis in document
        start = document.find(diagnosis)
        if start != -1:
            annotations.append((start, start + len(diagnosis), "DIAGNOSIS"))
        
        # Find medications in document
        start = document.find(medications)
        if start != -1:
            annotations.append((start, start + len(medications), "MEDICATION"))
        
        # Find doctor name in document
        if doctor_name in document:
            start = document.find(doctor_name)
            if start != -1:
                annotations.append((start, start + len(doctor_name), "NAME"))
        
        return document, annotations
    
    def generate_dataset(self, num_documents=100, train_ratio=0.7, val_ratio=0.15):
        """
        Generate a complete dataset with train/val/test splits.
        
        Args:
            num_documents: Total number of documents to generate
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            
        Returns:
            Dictionary with train/val/test splits, each containing texts and annotations
        """
        all_docs = []
        all_annotations = []
        
        # Generate documents
        for _ in range(num_documents):
            doc, anns = self.generate_synthetic_document()
            all_docs.append(doc)
            all_annotations.append(anns)
        
        # Split indices
        indices = list(range(num_documents))
        random.shuffle(indices)
        
        train_size = int(num_documents * train_ratio)
        val_size = int(num_documents * val_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create splits
        dataset = {
            "train": {
                "texts": [all_docs[i] for i in train_indices],
                "annotations": [all_annotations[i] for i in train_indices]
            },
            "val": {
                "texts": [all_docs[i] for i in val_indices],
                "annotations": [all_annotations[i] for i in val_indices]
            },
            "test": {
                "texts": [all_docs[i] for i in test_indices],
                "annotations": [all_annotations[i] for i in test_indices]
            }
        }
        
        return dataset
    
    def save_dataset(self, dataset, output_dir="data"):
        """
        Save a dataset to disk.
        
        Args:
            dataset: Dataset dictionary
            output_dir: Directory to save to
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for split in ["train", "val", "test"]:
            # Save texts
            with open(os.path.join(output_dir, f"{split}_texts.txt"), "w") as f:
                for text in dataset[split]["texts"]:
                    f.write(text)
                    f.write("\n---DOCUMENT_BOUNDARY---\n")
            
            # Save annotations
            with open(os.path.join(output_dir, f"{split}_annotations.jsonl"), "w") as f:
                for anns in dataset[split]["annotations"]:
                    f.write(json.dumps(anns))
                    f.write("\n")
    
    def load_dataset(self, input_dir="data"):
        """
        Load a dataset from disk.
        
        Args:
            input_dir: Directory to load from
            
        Returns:
            Dataset dictionary
        """
        dataset = {}
        
        for split in ["train", "val", "test"]:
            texts = []
            annotations = []
            
            # Load texts
            try:
                with open(os.path.join(input_dir, f"{split}_texts.txt"), "r") as f:
                    content = f.read()
                    documents = content.split("\n---DOCUMENT_BOUNDARY---\n")
                    texts = [doc for doc in documents if doc.strip()]
            except FileNotFoundError:
                print(f"Warning: {split}_texts.txt not found")
            
            # Load annotations
            try:
                with open(os.path.join(input_dir, f"{split}_annotations.jsonl"), "r") as f:
                    for line in f:
                        if line.strip():
                            anns = json.loads(line)
                            annotations.append(anns)
            except FileNotFoundError:
                print(f"Warning: {split}_annotations.jsonl not found")
            
            dataset[split] = {
                "texts": texts,
                "annotations": annotations
            }
        
        return dataset