#!/usr/bin/env python3
"""
Symptom extraction service for Healthcare Super-Assistant
Provides clinical NER for extracting symptoms, conditions, and medical entities from text
"""

import re
import json
import os
import asyncio
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
# Load .env for direct Python runs (optional)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass
from collections import defaultdict


class SymptomExtractor:
    """
    Clinical symptom and entity extractor using biomedical NER models
    """
    
    def __init__(self, model_name: str = "samrawal/bert-base-uncased_clinical-ner"):
        """
        Initialize the symptom extractor
        
        Args:
            model_name: Hugging Face model for clinical NER
        """
        self.model_name = model_name
        self.ner_pipeline = None
        self.nlp = None
        
        # Medical entity categories
        self.symptom_keywords = self._load_symptom_keywords()
        self.duration_patterns = self._load_duration_patterns()
        
        # Initialize models
        self._initialize_models()
        
        # Result cache (simple LRU) to avoid repeated NER on the same text
        self._result_cache = {}
        self._result_order = []
        try:
            self._result_cache_max = int(os.getenv("BAYMAX_NER_CACHE", "64"))
        except Exception:
            self._result_cache_max = 64
        
    def _initialize_models(self):
        """Initialize NER pipeline and spaCy model"""
        try:
            print(f"Loading clinical NER model: {self.model_name}")
            # Prefer GPU if available
            try:
                import torch  # type: ignore
                device = 0 if torch.cuda.is_available() else -1
            except Exception:
                device = -1
            self.ner_pipeline = pipeline(
                "ner", 
                model=self.model_name, 
                tokenizer=self.model_name,
                aggregation_strategy="simple",
                device=device
            )
            print("Clinical NER model loaded successfully")
        except Exception as e:
            print(f"Error loading NER model: {e}")
            print("Using fallback rule-based extraction")
            self.ner_pipeline = None
        
        # Skip spaCy loading for speed - not needed for basic extraction
        self.nlp = None
    
    def _load_symptom_keywords(self) -> Dict[str, List[str]]:
        """Load symptom keyword categories for rule-based extraction"""
        return {
            "pain": [
                "pain", "ache", "aching", "sore", "soreness", "hurt", "hurting",
                "sharp pain", "dull pain", "throbbing", "burning", "stabbing",
                "cramping", "tender", "tenderness"
            ],
            "fever": [
                "fever", "febrile", "high temperature", "hot", "chills", "shivering",
                "sweating", "night sweats", "feverish"
            ],
            "respiratory": [
                "cough", "coughing", "shortness of breath", "difficulty breathing",
                "breathless", "wheezing", "chest tightness", "congestion",
                "stuffy nose", "runny nose", "sore throat", "hoarse"
            ],
            "gastrointestinal": [
                "nausea", "vomiting", "diarrhea", "constipation", "bloating",
                "stomach pain", "abdominal pain", "indigestion", "heartburn",
                "loss of appetite", "appetite loss"
            ],
            "neurological": [
                "headache", "dizziness", "dizzy", "confusion", "memory loss",
                "numbness", "tingling", "weakness", "fatigue", "tired", "exhausted"
            ],
            "cardiovascular": [
                "chest pain", "palpitations", "irregular heartbeat", "swelling",
                "edema", "leg swelling", "ankle swelling"
            ],
            "dermatological": [
                "rash", "itching", "itchy", "redness", "swelling", "bruising",
                "lesion", "wound", "cut", "burn"
            ]
        }
    
    def _load_duration_patterns(self) -> List[str]:
        """Load regex patterns for extracting duration information"""
        return [
            r'(\d+)\s+(days?|weeks?|months?|years?)',
            r'(for|since|about|approximately|around)\s+(\d+)\s+(days?|weeks?|months?|years?)',
            r'(past|last)\s+(\d+)\s+(days?|weeks?|months?|years?)',
            r'(\d+)\s+(hours?|minutes?)',
            r'(few|several|couple)\s+(days?|weeks?|months?)',
            r'(yesterday|today|this morning|last night|recently|lately)'
        ]
    
    def extract_symptoms(self, text: str) -> Dict[str, any]:
        """
        Extract symptoms and medical entities from text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with extracted symptoms and metadata
        """
        # Cache hit
        key = (text or "").strip()
        if key and key in self._result_cache:
            return self._result_cache[key]

        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        
        # Extract using different methods
        ner_entities = self._extract_with_ner(cleaned_text) if self.ner_pipeline else []
        rule_based_symptoms = self._extract_with_rules(cleaned_text)
        duration_info = self._extract_duration(cleaned_text)
        severity_info = self._extract_severity(cleaned_text)
        
        # Combine and normalize results
        all_symptoms = self._combine_extractions(ner_entities, rule_based_symptoms)
        
        result = {
            "text": text,
            "cleaned_text": cleaned_text,
            "symptoms": all_symptoms,
            "duration": duration_info,
            "severity": severity_info,
            "entities_count": len(all_symptoms),
            "extraction_method": "hybrid" if self.ner_pipeline else "rule-based"
        }
        # Store in cache
        try:
            if key:
                self._result_cache[key] = result
                self._result_order.append(key)
                if len(self._result_order) > self._result_cache_max:
                    oldest = self._result_order.pop(0)
                    self._result_cache.pop(oldest, None)
        except Exception:
            pass
        return result
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize input text"""
        if not text:
            return ""
        
        # Convert to lowercase for processing (keep original for display)
        text = text.strip()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def _extract_with_ner(self, text: str) -> List[Dict]:
        """Extract entities using the NER model"""
        try:
            entities = self.ner_pipeline(text)
            
            processed_entities = []
            for entity in entities:
                processed_entities.append({
                    "text": entity.get("word", "").replace("##", ""),
                    "label": entity.get("entity_group", entity.get("entity", "UNKNOWN")),
                    "confidence": entity.get("score", 0.0),
                    "start": entity.get("start", 0),
                    "end": entity.get("end", 0),
                    "method": "ner"
                })
            
            return processed_entities
            
        except Exception as e:
            print(f"Error in NER extraction: {e}")
            return []
    
    def _extract_with_rules(self, text: str) -> List[Dict]:
        """Extract symptoms using rule-based keyword matching"""
        text_lower = text.lower()
        found_symptoms = []
        
        for category, keywords in self.symptom_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    # Find the position in original text
                    start_pos = text_lower.find(keyword.lower())
                    if start_pos != -1:
                        found_symptoms.append({
                            "text": keyword,
                            "label": f"SYMPTOM_{category.upper()}",
                            "confidence": 0.8,  # Rule-based confidence
                            "start": start_pos,
                            "end": start_pos + len(keyword),
                            "category": category,
                            "method": "rule-based"
                        })
        
        return found_symptoms
    
    def _extract_duration(self, text: str) -> List[Dict]:
        """Extract duration/temporal information"""
        duration_info = []
        
        for pattern in self.duration_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                duration_info.append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "pattern": pattern
                })
        
        return duration_info
    
    def _extract_severity(self, text: str) -> List[Dict]:
        """Extract severity indicators"""
        severity_keywords = {
            "mild": ["mild", "slight", "minor", "little", "small"],
            "moderate": ["moderate", "medium", "some", "noticeable"],
            "severe": ["severe", "intense", "extreme", "terrible", "awful", 
                      "unbearable", "excruciating", "very", "really", "extremely"]
        }
        
        text_lower = text.lower()
        severity_info = []
        
        for severity, keywords in severity_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    start_pos = text_lower.find(keyword)
                    if start_pos != -1:
                        severity_info.append({
                            "text": keyword,
                            "severity": severity,
                            "confidence": 0.7,
                            "start": start_pos,
                            "end": start_pos + len(keyword)
                        })
        
        return severity_info
    
    def _combine_extractions(self, ner_entities: List[Dict], rule_entities: List[Dict]) -> List[Dict]:
        """Combine and deduplicate entities from different extraction methods"""
        all_entities = []
        
        # Add NER entities
        for entity in ner_entities:
            all_entities.append(entity)
        
        # Add rule-based entities, avoiding duplicates
        for rule_entity in rule_entities:
            # Check for overlap with existing entities
            is_duplicate = False
            for existing in all_entities:
                if (abs(existing["start"] - rule_entity["start"]) < 5 and 
                    existing["text"].lower() in rule_entity["text"].lower()):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                all_entities.append(rule_entity)
        
        # Sort by position in text
        all_entities.sort(key=lambda x: x["start"])
        
        return all_entities
    
    async def extract_symptoms_async(self, text: str) -> Dict[str, any]:
        """Async wrapper to run extraction off the main thread."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.extract_symptoms, text)

    def normalize_symptoms(self, symptoms: List[Dict]) -> Dict[str, List[str]]:
        """Normalize extracted symptoms into standard categories"""
        normalized = defaultdict(list)
        
        for symptom in symptoms:
            text = symptom["text"].lower()
            label = symptom.get("label", "").lower()
            category = symptom.get("category", "").lower()
            
            # Map to standard categories
            if any(keyword in text for keyword in ["pain", "ache", "hurt", "sore"]):
                normalized["pain"].append(symptom["text"])
            elif any(keyword in text for keyword in ["fever", "temperature", "chills"]):
                normalized["fever"].append(symptom["text"])
            elif any(keyword in text for keyword in ["cough", "breath", "wheez"]):
                normalized["respiratory"].append(symptom["text"])
            elif any(keyword in text for keyword in ["nausea", "vomit", "diarrhea", "stomach"]):
                normalized["gastrointestinal"].append(symptom["text"])
            elif category:
                normalized[category].append(symptom["text"])
            else:
                normalized["other"].append(symptom["text"])
        
        # Remove duplicates
        for category in normalized:
            normalized[category] = list(set(normalized[category]))
        
        return dict(normalized)
    
    def health_check(self) -> Dict[str, any]:
        """Check if symptom extractor is working correctly"""
        try:
            test_text = "I have a headache and fever for 3 days"
            result = self.extract_symptoms(test_text)
            
            return {
                "status": "healthy",
                "ner_model_loaded": self.ner_pipeline is not None,
                "spacy_loaded": self.nlp is not None,
                "test_extraction_count": result["entities_count"],
                "extraction_method": result["extraction_method"]
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


def create_symptom_extractor(model_name: Optional[str] = None) -> SymptomExtractor:
    """
    Convenience function to create a symptom extractor
    
    Args:
        model_name: Optional custom model name
        
    Returns:
        Configured SymptomExtractor instance
    """
    # Allow environment override for faster/lighter models without code changes
    env_model = os.getenv("BAYMAX_NER_MODEL", "").strip()
    effective_model = model_name or (env_model if env_model else None)
    if effective_model:
        return SymptomExtractor(model_name=effective_model)
    else:
        return SymptomExtractor()


# Example usage and testing
if __name__ == "__main__":
    print("Healthcare Super-Assistant - Symptom Extraction Test")
    print("=" * 58)
    
    try:
        # Create extractor
        extractor = create_symptom_extractor()
        
        # Health check
        health = extractor.health_check()
        print(f"Health check: {health['status']}")
        print(f"NER model loaded: {health['ner_model_loaded']}")
        print(f"Extraction method: {health['extraction_method']}")
        
        # Test with sample texts
        test_texts = [
            "I have been having severe headaches and nausea for the past 3 days",
            "Patient reports chest pain and shortness of breath since yesterday",
            "Experiencing mild fever, cough, and fatigue for about a week",
            "Sharp abdominal pain with vomiting started this morning"
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\nTest {i}: '{text}'")
            result = extractor.extract_symptoms(text)
            
            print(f"  Entities found: {result['entities_count']}")
            print(f"  Method: {result['extraction_method']}")
            
            # Show extracted symptoms
            if result['symptoms']:
                print("  Symptoms:")
                for symptom in result['symptoms'][:3]:  # Show first 3
                    print(f"    - {symptom['text']} ({symptom['label']}) - {symptom['confidence']:.2f}")
            
            # Show normalized categories
            normalized = extractor.normalize_symptoms(result['symptoms'])
            if normalized:
                print("  Categories:", list(normalized.keys()))
            
            # Show duration if found
            if result['duration']:
                duration_texts = [d['text'] for d in result['duration']]
                print(f"  Duration: {', '.join(duration_texts)}")
        
        print("\nSymptom extraction testing completed!")
        
    except Exception as e:
        print(f"Error testing symptom extractor: {e}")
        import traceback
        traceback.print_exc()
