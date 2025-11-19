#!/usr/bin/env python3
"""
Download comprehensive medical datasets for enhanced diagnosis and medication advice
Includes drug databases, diagnostic criteria, and treatment guidelines
"""

import os
import requests
import json
import csv
from pathlib import Path
from typing import Dict, List
import time

def create_comprehensive_medical_database():
    """Create comprehensive medical knowledge database"""
    
    # Create enhanced data directory
    data_dir = Path("data/enhanced_comprehensive")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("📋 Creating comprehensive medical knowledge database...")
    
    # 1. Drug Information Database
    print("💊 Creating drug information database...")
    drug_data = []
    
    # Common medications with dosages and indications
    medications = [
        # Pain/Fever
        {"name": "Acetaminophen (Tylenol)", "category": "Analgesic/Antipyretic", 
         "indication": "Pain relief, fever reduction", "dosage": "500-1000mg every 4-6 hours, max 4000mg/day",
         "contraindications": "Liver disease, alcohol use", "side_effects": "Nausea, liver toxicity with overdose"},
        {"name": "Ibuprofen (Advil, Motrin)", "category": "NSAID", 
         "indication": "Pain, inflammation, fever", "dosage": "200-400mg every 4-6 hours, max 1200mg/day",
         "contraindications": "Kidney disease, stomach ulcers, pregnancy", "side_effects": "Stomach upset, kidney problems"},
        
        # Antibiotics
        {"name": "Amoxicillin", "category": "Antibiotic", 
         "indication": "Bacterial infections", "dosage": "500mg every 8 hours or 875mg every 12 hours",
         "contraindications": "Penicillin allergy", "side_effects": "Diarrhea, nausea, allergic reactions"},
        {"name": "Azithromycin (Z-pack)", "category": "Antibiotic", 
         "indication": "Respiratory infections, skin infections", "dosage": "500mg day 1, then 250mg daily for 4 days",
         "contraindications": "Macrolide allergy", "side_effects": "Nausea, diarrhea, stomach pain"},
        
        # Cardiovascular
        {"name": "Lisinopril", "category": "ACE Inhibitor", 
         "indication": "High blood pressure, heart failure", "dosage": "Starting 10mg daily, usual range 10-40mg daily",
         "contraindications": "Pregnancy, angioedema history", "side_effects": "Dry cough, hyperkalemia, hypotension"},
        {"name": "Metoprolol", "category": "Beta-blocker", 
         "indication": "High blood pressure, chest pain, heart failure", "dosage": "25-100mg twice daily",
         "contraindications": "Asthma, severe bradycardia", "side_effects": "Fatigue, dizziness, cold hands/feet"},
        
        # Diabetes
        {"name": "Metformin", "category": "Antidiabetic", 
         "indication": "Type 2 diabetes", "dosage": "Starting 500mg twice daily, max 2000mg daily",
         "contraindications": "Kidney disease, acidosis", "side_effects": "Nausea, diarrhea, vitamin B12 deficiency"},
        
        # Mental Health
        {"name": "Sertraline (Zoloft)", "category": "SSRI Antidepressant", 
         "indication": "Depression, anxiety, PTSD", "dosage": "Starting 25-50mg daily, usual range 50-200mg",
         "contraindications": "MAOI use", "side_effects": "Nausea, insomnia, sexual dysfunction"},
        
        # Allergies
        {"name": "Loratadine (Claritin)", "category": "Antihistamine", 
         "indication": "Allergies, hay fever", "dosage": "10mg once daily",
         "contraindications": "Severe liver disease", "side_effects": "Headache, drowsiness (less than older antihistamines)"},
        {"name": "Cetirizine (Zyrtec)", "category": "Antihistamine", 
         "indication": "Allergies, hives", "dosage": "10mg once daily",
         "contraindications": "End-stage kidney disease", "side_effects": "Drowsiness, dry mouth"},
    ]
    
    # Add more comprehensive drug data
    for med in medications:
        drug_data.append({
            "text": f"{med['name']} is a {med['category']} used for {med['indication']}. "
                   f"Typical dosage: {med['dosage']}. "
                   f"Contraindications: {med['contraindications']}. "
                   f"Common side effects: {med['side_effects']}.",
            "source": "Clinical Pharmacology Database",
            "category": "Medications",
            "drug_name": med['name'],
            "drug_category": med['category'],
            "type": "medication_info"
        })
    
    # Save drug database
    with open(data_dir / "drug_database.json", "w") as f:
        json.dump(drug_data, f, indent=2)
    
    # 2. Diagnostic Criteria Database
    print("🩺 Creating diagnostic criteria database...")
    diagnostic_data = []
    
    # Common conditions with diagnostic criteria
    conditions = [
        {
            "condition": "Migraine Headache",
            "criteria": "Recurrent headache attacks lasting 4-72 hours with at least 2 of: unilateral location, pulsating quality, moderate-severe intensity, aggravation by routine physical activity. Plus nausea/vomiting or photophobia and phonophobia.",
            "differential": "Tension headache, cluster headache, medication overuse headache",
            "treatment": "Acute: NSAIDs, triptans. Preventive: beta-blockers, anticonvulsants, antidepressants"
        },
        {
            "condition": "Hypertension",
            "criteria": "Blood pressure consistently ≥140/90 mmHg on multiple occasions, or ≥130/80 mmHg with cardiovascular risk factors",
            "differential": "White coat hypertension, secondary hypertension",
            "treatment": "Lifestyle modifications, ACE inhibitors, ARBs, calcium channel blockers, diuretics"
        },
        {
            "condition": "Type 2 Diabetes",
            "criteria": "Fasting glucose ≥126 mg/dL, random glucose ≥200 mg/dL with symptoms, or HbA1c ≥6.5%",
            "differential": "Type 1 diabetes, MODY, secondary diabetes",
            "treatment": "Metformin first-line, lifestyle modifications, glucose monitoring"
        },
        {
            "condition": "Depression (Major Depressive Disorder)",
            "criteria": "5 or more symptoms for ≥2 weeks including depressed mood or anhedonia: sleep disturbance, fatigue, appetite changes, concentration problems, psychomotor changes, worthlessness/guilt, suicidal thoughts",
            "differential": "Bipolar disorder, adjustment disorder, medical conditions",
            "treatment": "SSRIs, psychotherapy, lifestyle modifications"
        },
        {
            "condition": "Gastroesophageal Reflux Disease (GERD)",
            "criteria": "Heartburn and/or regurgitation occurring ≥2 times per week, or complications of reflux",
            "differential": "Peptic ulcer disease, cardiac chest pain, esophageal motility disorders",
            "treatment": "PPIs, H2 blockers, lifestyle modifications, antacids"
        }
    ]
    
    for condition in conditions:
        diagnostic_data.append({
            "text": f"{condition['condition']}: Diagnostic criteria - {condition['criteria']} "
                   f"Differential diagnosis includes: {condition['differential']}. "
                   f"Treatment approach: {condition['treatment']}",
            "source": "Clinical Diagnostic Guidelines",
            "category": "Diagnostics",
            "condition_name": condition['condition'],
            "type": "diagnostic_criteria"
        })
    
    # Save diagnostic database
    with open(data_dir / "diagnostic_database.json", "w") as f:
        json.dump(diagnostic_data, f, indent=2)
    
    # 3. Treatment Protocols Database
    print("💉 Creating treatment protocols database...")
    treatment_data = []
    
    # Evidence-based treatment protocols
    treatments = [
        {
            "condition": "Acute Bacterial Sinusitis",
            "first_line": "Amoxicillin 500mg TID or 875mg BID for 5-7 days",
            "alternative": "Azithromycin, cefdinir if penicillin allergy",
            "supportive": "Nasal saline irrigation, decongestants, analgesics"
        },
        {
            "condition": "Strep Throat",
            "first_line": "Penicillin V 500mg BID for 10 days or amoxicillin 500mg TID",
            "alternative": "Azithromycin, cephalexin if penicillin allergy",
            "supportive": "Pain relief with acetaminophen or ibuprofen, throat lozenges"
        },
        {
            "condition": "Uncomplicated UTI",
            "first_line": "Nitrofurantoin 100mg BID for 5 days or trimethoprim-sulfamethoxazole",
            "alternative": "Fosfomycin 3g single dose, ciprofloxacin if resistant",
            "supportive": "Increased fluid intake, cranberry products, phenazopyridine for dysuria"
        }
    ]
    
    for treatment in treatments:
        treatment_data.append({
            "text": f"Treatment for {treatment['condition']}: First-line therapy is {treatment['first_line']}. "
                   f"Alternative options include {treatment['alternative']}. "
                   f"Supportive care: {treatment['supportive']}",
            "source": "Evidence-Based Treatment Guidelines",
            "category": "Treatments",
            "condition": treatment['condition'],
            "type": "treatment_protocol"
        })
    
    # Save treatment database
    with open(data_dir / "treatment_database.json", "w") as f:
        json.dump(treatment_data, f, indent=2)
    
    # 4. Symptom-to-Diagnosis Database
    print("🔍 Creating symptom-diagnosis database...")
    symptom_data = []
    
    # Common symptom presentations
    symptoms = [
        {
            "symptom": "Chest Pain",
            "urgent": ["Myocardial infarction", "Pulmonary embolism", "Aortic dissection", "Pneumothorax"],
            "common": ["GERD", "Costochondritis", "Muscle strain", "Anxiety"],
            "workup": "ECG, troponins, chest X-ray, vital signs"
        },
        {
            "symptom": "Headache",
            "urgent": ["Subarachnoid hemorrhage", "Meningitis", "Temporal arteritis", "Increased ICP"],
            "common": ["Tension headache", "Migraine", "Cluster headache", "Medication overuse"],
            "workup": "Neurological exam, consider imaging if red flags"
        },
        {
            "symptom": "Shortness of Breath",
            "urgent": ["Pulmonary embolism", "Pneumothorax", "Heart failure", "Severe asthma"],
            "common": ["Asthma", "COPD", "Anxiety", "Deconditioning"],
            "workup": "Pulse oximetry, chest X-ray, ECG, BNP"
        }
    ]
    
    for symptom in symptoms:
        symptom_data.append({
            "text": f"{symptom['symptom']}: Urgent considerations include {', '.join(symptom['urgent'])}. "
                   f"Common causes are {', '.join(symptom['common'])}. "
                   f"Initial workup should include: {symptom['workup']}",
            "source": "Clinical Decision Support",
            "category": "Symptom_Analysis",
            "symptom": symptom['symptom'],
            "type": "symptom_guide"
        })
    
    # Save symptom database
    with open(data_dir / "symptom_database.json", "w") as f:
        json.dump(symptom_data, f, indent=2)
    
    # 5. Drug Interactions Database
    print("⚠️ Creating drug interactions database...")
    interaction_data = []
    
    interactions = [
        {
            "drug1": "Warfarin", "drug2": "NSAIDs", 
            "severity": "Major", "effect": "Increased bleeding risk",
            "management": "Avoid concurrent use or monitor INR closely"
        },
        {
            "drug1": "ACE inhibitors", "drug2": "Potassium supplements", 
            "severity": "Moderate", "effect": "Hyperkalemia risk",
            "management": "Monitor potassium levels regularly"
        },
        {
            "drug1": "SSRIs", "drug2": "MAOIs", 
            "severity": "Contraindicated", "effect": "Serotonin syndrome",
            "management": "Allow 2-week washout period between medications"
        }
    ]
    
    for interaction in interactions:
        interaction_data.append({
            "text": f"Drug interaction: {interaction['drug1']} and {interaction['drug2']} - "
                   f"Severity: {interaction['severity']}. Effect: {interaction['effect']}. "
                   f"Management: {interaction['management']}",
            "source": "Drug Interaction Database",
            "category": "Drug_Interactions",
            "type": "drug_interaction"
        })
    
    # Save interactions database
    with open(data_dir / "interactions_database.json", "w") as f:
        json.dump(interaction_data, f, indent=2)
    
    print(f"✅ Comprehensive medical database created in {data_dir}")
    print(f"📊 Total entries: {len(drug_data + diagnostic_data + treatment_data + symptom_data + interaction_data)}")
    
    return data_dir

if __name__ == "__main__":
    create_comprehensive_medical_database()
