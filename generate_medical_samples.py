#!/usr/bin/env python
"""
Generate synthetic medical samples for PHI/PII detection testing.
This avoids scraping issues by creating realistic medical records with PHI.
"""
import os
import json
import random
import argparse
from datetime import datetime, timedelta


class MedicalSampleGenerator:
    """Generate synthetic medical documents with PHI."""
    
    def __init__(self):
        """Initialize generator with sample data."""
        # Patient data components
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
        
        self.street_names = [
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
        
        self.states = {
            "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", 
            "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
            "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
            "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
            "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
            "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
            "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada"
        }
        
        self.hospitals = [
            "General Hospital", "Memorial Hospital", "University Medical Center",
            "Community Hospital", "Regional Medical Center", "Methodist Hospital",
            "St. Mary's Hospital", "County Hospital", "Mercy Medical Center",
            "Veterans Affairs Medical Center", "Children's Hospital", "Medical Center"
        ]

        self.doctor_specialties = [
            "Cardiology", "Neurology", "Oncology", "Pediatrics", "Internal Medicine",
            "Family Medicine", "Obstetrics", "Dermatology", "Orthopedics", "Psychiatry",
            "Radiology", "Urology", "Gastroenterology", "Endocrinology", "Rheumatology"
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
        
        self.document_types = ["Discharge Summary", "Progress Note", "Consult Note", 
                             "Emergency Room Report", "History and Physical", "Soap Note",
                             "Radiology Report", "Operative Report"]

    def generate_patient(self):
        """Generate a complete patient profile with PHI."""
        # Basic demographics
        gender = random.choice(["Male", "Female"])
        first_name = random.choice(self.first_names)
        last_name = random.choice(self.last_names)
        
        # Generate birthdate (adult patient)
        years = random.randint(18, 85)
        birthdate = datetime.now() - timedelta(days=365 * years + random.randint(0, 364))
        dob = birthdate.strftime("%m/%d/%Y")
        age = years  # Simplified age calculation
        
        # Generate address
        street_num = random.randint(100, 9999)
        street = random.choice(self.street_names)
        city = random.choice(self.cities)
        state_abbr = random.choice(list(self.states.keys()))
        state = self.states[state_abbr]
        zipcode = f"{random.randint(10000, 99999)}"
        
        # Generate contact info
        phone = f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}"
        email = f"{first_name.lower()}.{last_name.lower()}@{random.choice(['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com'])}"
        
        # Generate medical identifiers
        mrn = f"{random.randint(1000000, 9999999)}"
        ssn = f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
        insurance_id = f"{random.choice(['BC', 'UHC', 'AETNA', 'CIGNA'])}{random.randint(10000000, 99999999)}"
        
        # Build patient object
        patient = {
            "name": {
                "first": first_name,
                "last": last_name,
                "full": f"{first_name} {last_name}"
            },
            "gender": gender,
            "birth_date": dob,
            "age": age,
            "address": {
                "street": f"{street_num} {street}",
                "city": city,
                "state": state,
                "state_abbr": state_abbr,
                "zip": zipcode,
                "full": f"{street_num} {street}, {city}, {state_abbr} {zipcode}"
            },
            "contact": {
                "phone": phone,
                "email": email
            },
            "identifiers": {
                "mrn": mrn,
                "ssn": ssn,
                "insurance_id": insurance_id
            }
        }
        
        return patient
    
    def generate_provider(self):
        """Generate a healthcare provider profile."""
        gender = random.choice(["Male", "Female"])
        first_name = random.choice(self.first_names)
        last_name = random.choice(self.last_names)
        
        specialty = random.choice(self.doctor_specialties)
        credentials = random.choice(["MD", "DO", "NP", "PA"])
        
        hospital = random.choice(self.hospitals)
        department = specialty
        
        phone = f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}"
        email = f"{first_name.lower()}.{last_name.lower()}@{hospital.lower().replace(' ', '')}.org"
        
        provider = {
            "name": {
                "first": first_name,
                "last": last_name,
                "full": f"{first_name} {last_name}, {credentials}"
            },
            "specialty": specialty,
            "credentials": credentials,
            "facility": {
                "name": hospital,
                "department": department
            },
            "contact": {
                "phone": phone,
                "email": email
            }
        }
        
        return provider
    
    def generate_clinical_info(self):
        """Generate clinical information for a medical document."""
        # Select diagnoses and medications
        num_diagnoses = random.randint(1, 3)
        num_medications = random.randint(1, 5)
        
        diagnoses = random.sample(self.diagnoses, num_diagnoses)
        medications = random.sample(self.medications, num_medications)
        
        # Generate dates for clinical events
        admission_date = datetime.now() - timedelta(days=random.randint(5, 30))
        admission_date_str = admission_date.strftime("%m/%d/%Y")
        
        discharge_date = admission_date + timedelta(days=random.randint(1, 10))
        discharge_date_str = discharge_date.strftime("%m/%d/%Y")
        
        # Generate vital signs
        vitals = {
            "temperature": round(random.uniform(97.0, 99.5), 1),
            "heart_rate": random.randint(60, 100),
            "blood_pressure": f"{random.randint(110, 140)}/{random.randint(70, 90)}",
            "respiratory_rate": random.randint(12, 20),
            "oxygen_saturation": random.randint(95, 100)
        }
        
        clinical_info = {
            "diagnoses": diagnoses,
            "medications": medications,
            "dates": {
                "admission": admission_date_str,
                "discharge": discharge_date_str
            },
            "vitals": vitals
        }
        
        return clinical_info
    
    def generate_discharge_summary(self, patient, provider, clinical_info):
        """Generate a discharge summary document."""
        # Format the document
        document = f"""
DISCHARGE SUMMARY

PATIENT: {patient['name']['full']}
MRN: {patient['identifiers']['mrn']}
DOB: {patient['birth_date']} ({patient['age']} years)
ADMISSION DATE: {clinical_info['dates']['admission']}
DISCHARGE DATE: {clinical_info['dates']['discharge']}
ATTENDING PHYSICIAN: {provider['name']['full']}

DISCHARGE DIAGNOSES:
{chr(10).join('- ' + d for d in clinical_info['diagnoses'])}

HISTORY OF PRESENT ILLNESS:
{patient['name']['first']} {patient['name']['last']} is a {patient['age']}-year-old {patient['gender'].lower()} with a history of {clinical_info['diagnoses'][0].lower()} who presented to {provider['facility']['name']} on {clinical_info['dates']['admission']} with complaints of {random.choice(['shortness of breath', 'chest pain', 'abdominal pain', 'fever', 'fatigue'])}. The patient reported symptoms began approximately {random.randint(1, 7)} days prior to admission. 

HOSPITAL COURSE:
The patient was admitted to the {provider['facility']['department']} service. Initial vital signs showed temperature {clinical_info['vitals']['temperature']}°F, heart rate {clinical_info['vitals']['heart_rate']} bpm, blood pressure {clinical_info['vitals']['blood_pressure']} mmHg, respiratory rate {clinical_info['vitals']['respiratory_rate']}/min, and oxygen saturation {clinical_info['vitals']['oxygen_saturation']}% on room air.

Laboratory tests revealed {random.choice(['elevated white blood cell count', 'normal complete blood count', 'mild anemia', 'elevated liver enzymes', 'normal renal function'])}. The patient was treated with {random.choice(['IV antibiotics', 'IV fluids', 'pain management', 'oxygen therapy', 'anti-inflammatory medications'])}.

During hospitalization, the patient {random.choice(['improved steadily', 'responded well to treatment', 'had an uncomplicated hospital course', 'experienced mild complications that resolved', 'required minimal intervention'])}.

DISCHARGE MEDICATIONS:
{chr(10).join('- ' + m + ' ' + random.choice(['10mg daily', '20mg twice daily', '5mg as needed', '100mg every 8 hours', '15ml twice daily']) for m in clinical_info['medications'])}

DISCHARGE INSTRUCTIONS:
The patient is to follow up with Dr. {provider['name']['last']} in {random.randint(1, 4)} weeks. Call for any fever greater than 101°F, increasing pain, or new symptoms. Resume regular diet and activity as tolerated.

CONTACT INFORMATION:
If you have any questions, please contact us at {provider['contact']['phone']}.

PATIENT ADDRESS:
{patient['address']['full']}

PATIENT PHONE:
{patient['contact']['phone']}

PATIENT EMAIL:
{patient['contact']['email']}

SSN: {patient['identifiers']['ssn']}
Insurance ID: {patient['identifiers']['insurance_id']}

Electronically signed by:
{provider['name']['full']}
{provider['specialty']}
{provider['facility']['name']}
{provider['contact']['email']}
"""
        
        return document.strip()
    
    def generate_progress_note(self, patient, provider, clinical_info):
        """Generate a progress note document."""
        document = f"""
PROGRESS NOTE

PATIENT: {patient['name']['full']}
MRN: {patient['identifiers']['mrn']}
DOB: {patient['birth_date']}
DATE OF SERVICE: {datetime.now().strftime("%m/%d/%Y")}
PROVIDER: {provider['name']['full']}

SUBJECTIVE:
{patient['name']['first']} is a {patient['age']}-year-old {patient['gender'].lower()} with {clinical_info['diagnoses'][0].lower()} presenting for follow-up. Patient reports {random.choice(['feeling better', 'improved symptoms', 'no new concerns', 'mild persistent symptoms', 'good response to medications'])}. {random.choice(['No side effects from medications.', 'Tolerating medications well.', 'Some minor side effects from medications.', 'Concerned about medication costs.', 'Requests medication refills.'])}

The patient resides at {patient['address']['full']} and can be reached at {patient['contact']['phone']} or {patient['contact']['email']}.

OBJECTIVE:
Vital Signs: Temperature {clinical_info['vitals']['temperature']}°F, heart rate {clinical_info['vitals']['heart_rate']} bpm, blood pressure {clinical_info['vitals']['blood_pressure']} mmHg, respiratory rate {clinical_info['vitals']['respiratory_rate']}/min, oxygen saturation {clinical_info['vitals']['oxygen_saturation']}%.

Physical Examination:
General: {random.choice(['Alert and oriented', 'Well-appearing', 'No acute distress', 'Mildly uncomfortable', 'Comfortable'])}
HEENT: {random.choice(['Normocephalic, atraumatic', 'Moist mucous membranes', 'No sinus tenderness', 'Clear oropharynx', 'No lesions noted'])}
Cardiovascular: {random.choice(['Regular rate and rhythm', 'No murmurs', 'Normal S1/S2', 'No gallops or rubs', 'Strong peripheral pulses'])}
Respiratory: {random.choice(['Clear to auscultation bilaterally', 'No wheezes or rhonchi', 'Good air movement', 'No respiratory distress', 'Normal breath sounds'])}
Abdomen: {random.choice(['Soft, non-tender', 'No hepatosplenomegaly', 'Normal bowel sounds', 'No guarding or rebound', 'No masses palpated'])}

ASSESSMENT AND PLAN:
1. {clinical_info['diagnoses'][0]}: {random.choice(['Stable', 'Improved', 'Well-controlled', 'Responding to treatment', 'Requires medication adjustment'])}. Will continue current management.
{('2. ' + clinical_info['diagnoses'][1] + ': ' + random.choice(['Stable', 'Improved', 'Well-controlled', 'Responding to treatment', 'Requires medication adjustment']) + '.') if len(clinical_info['diagnoses']) > 1 else ''}

Medications:
{chr(10).join('- ' + m + ' ' + random.choice(['10mg daily', '20mg twice daily', '5mg as needed', '100mg every 8 hours', '15ml twice daily']) for m in clinical_info['medications'])}

Patient instructed to follow up in {random.randint(1, 6)} months.

SOCIAL HISTORY:
SSN: {patient['identifiers']['ssn']}
Insurance: {patient['identifiers']['insurance_id']}

Electronically signed by:
{provider['name']['full']}
{provider['specialty']}
{provider['facility']['name']}
{provider['contact']['email']}
"""
        
        return document.strip()
    
    def generate_document(self, doc_type=None):
        """Generate a complete medical document with annotations."""
        # Generate components
        patient = self.generate_patient()
        provider = self.generate_provider()
        clinical_info = self.generate_clinical_info()
        
        # Choose document type if not specified
        if not doc_type:
            doc_type = random.choice(self.document_types)
        
        # Generate document based on type
        if doc_type == "Discharge Summary":
            document = self.generate_discharge_summary(patient, provider, clinical_info)
        else:  # Default to progress note for now
            document = self.generate_progress_note(patient, provider, clinical_info)
        
        # Generate annotations by finding PHI in the document
        annotations = []
        
        # Patient name annotations
        full_name = patient['name']['full']
        first_name = patient['name']['first']
        last_name = patient['name']['last']
        
        for name in [full_name, first_name, last_name]:
            start = 0
            while True:
                start = document.find(name, start)
                if start == -1:
                    break
                annotations.append((start, start + len(name), "NAME"))
                start += len(name)
        
        # Other PHI annotations
        phi_items = [
            (patient['identifiers']['mrn'], "MEDICAL_RECORD"),
            (patient['identifiers']['ssn'], "SSN"),
            (patient['identifiers']['insurance_id'], "ID"),
            (patient['birth_date'], "DATE"),
            (patient['contact']['phone'], "PHONE"),
            (patient['contact']['email'], "EMAIL"),
            (patient['address']['full'], "ADDRESS"),
            (patient['address']['street'], "ADDRESS"),
            (provider['name']['full'], "NAME"),
            (provider['contact']['email'], "EMAIL"),
            (provider['contact']['phone'], "PHONE")
        ]
        
        for item, phi_type in phi_items:
            start = 0
            while True:
                start = document.find(item, start)
                if start == -1:
                    break
                annotations.append((start, start + len(item), phi_type))
                start += len(item)
        
        # Sort annotations by start position
        annotations.sort(key=lambda x: x[0])
        
        return {
            "document": document,
            "annotations": annotations,
            "metadata": {
                "document_type": doc_type,
                "patient": patient,
                "provider": provider,
                "clinical_info": clinical_info
            }
        }
    
    def generate_dataset(self, num_documents=50, output_dir="medical_samples"):
        """Generate a dataset of medical documents with annotations."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        all_documents = []
        
        for i in range(num_documents):
            # Generate document
            doc_type = random.choice(self.document_types)
            document_data = self.generate_document(doc_type)
            
            # Save document
            filename = f"{doc_type.lower().replace(' ', '_')}_{i+1}.txt"
            with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
                f.write(document_data["document"])
            
            # Save annotations
            ann_filename = f"{doc_type.lower().replace(' ', '_')}_{i+1}.ann"
            with open(os.path.join(output_dir, ann_filename), "w", encoding="utf-8") as f:
                json.dump(document_data["annotations"], f, indent=2)
            
            # Save metadata
            meta_filename = f"{doc_type.lower().replace(' ', '_')}_{i+1}_meta.json"
            with open(os.path.join(output_dir, meta_filename), "w", encoding="utf-8") as f:
                json.dump(document_data["metadata"], f, indent=2)
            
            all_documents.append({
                "filename": filename,
                "annotation_file": ann_filename,
                "metadata_file": meta_filename,
                "document_type": doc_type,
                "num_annotations": len(document_data["annotations"])
            })
            
            print(f"Generated document {i+1}/{num_documents}: {filename}")
        
        # Save dataset index
        index_file = os.path.join(output_dir, "dataset_index.json")
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(all_documents, f, indent=2)
        
        # Create splits for train/val/test
        random.shuffle(all_documents)
        train_split = int(len(all_documents) * 0.7)
        val_split = int(len(all_documents) * 0.85)
        
        splits = {
            "train": [doc["filename"] for doc in all_documents[:train_split]],
            "val": [doc["filename"] for doc in all_documents[train_split:val_split]],
            "test": [doc["filename"] for doc in all_documents[val_split:]]
        }
        
        splits_file = os.path.join(output_dir, "splits.json")
        with open(splits_file, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)
        
        print(f"\nDataset generation complete:")
        print(f"  Total documents: {len(all_documents)}")
        print(f"  Training documents: {len(splits['train'])}")
        print(f"  Validation documents: {len(splits['val'])}")
        print(f"  Test documents: {len(splits['test'])}")
        print(f"  Files saved to: {output_dir}")
        
        return output_dir


def main():
    """Parse arguments and generate medical dataset."""
    parser = argparse.ArgumentParser(description="Generate synthetic medical documents with PHI")
    
    parser.add_argument("--num_documents", type=int, default=50,
                      help="Number of documents to generate")
    parser.add_argument("--output_dir", type=str, default="medical_samples",
                      help="Directory to save generated documents")
    
    args = parser.parse_args()
    
    print(f"Generating {args.num_documents} synthetic medical documents...")
    
    generator = MedicalSampleGenerator()
    output_dir = generator.generate_dataset(args.num_documents, args.output_dir)
    
    print(f"\nTo run PHI detection on this dataset:")
    print(f"./load_custom_data.py --input {output_dir} --input_type directory")


if __name__ == "__main__":
    main()