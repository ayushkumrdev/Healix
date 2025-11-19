#!/usr/bin/env python3
"""
Safety features and evaluation system for Healthcare Super-Assistant
Provides audit logging, response evaluation, and safety monitoring
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict, Counter


class SafetyMonitor:
    """
    Safety monitoring and audit logging for healthcare AI system
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize safety monitor
        
        Args:
            log_dir: Directory to store audit logs and safety data
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self.logger = self._setup_logger()
        
        # Safety thresholds
        self.safety_config = {
            "min_confidence_threshold": 0.3,
            "emergency_keywords_threshold": 1,
            "max_queries_per_session": 50,
            "suspicious_pattern_threshold": 5
        }
        
        # Track session statistics
        self.session_stats = defaultdict(int)
        self.response_history = []
        
        # Load emergency patterns
        self.emergency_patterns = self._load_emergency_patterns()
        self.safety_violations = []
        
        self.logger.info("SafetyMonitor initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup audit logging"""
        logger = logging.getLogger('baymax_audit')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        if not logger.handlers:
            logger.addHandler(file_handler)
        
        return logger
    
    def _load_emergency_patterns(self) -> Dict[str, List[str]]:
        """Load emergency detection patterns"""
        return {
            "cardiac": [
                "chest pain", "heart attack", "cardiac arrest", "palpitations",
                "irregular heartbeat", "heart pounding"
            ],
            "respiratory": [
                "difficulty breathing", "shortness of breath", "can't breathe",
                "wheezing", "gasping", "choking"
            ],
            "neurological": [
                "stroke", "seizure", "unconscious", "unresponsive", "paralysis",
                "severe headache", "confusion", "memory loss"
            ],
            "trauma": [
                "severe bleeding", "heavy bleeding", "major injury", "trauma",
                "broken bone", "head injury", "burns"
            ],
            "allergic": [
                "anaphylaxis", "severe allergic reaction", "swelling",
                "hives", "difficulty swallowing"
            ],
            "mental_health": [
                "suicide", "self harm", "want to die", "end it all",
                "kill myself", "suicidal thoughts"
            ]
        }
    
    def log_interaction(self, 
                       user_query: str, 
                       response_data: Dict[str, Any],
                       session_id: Optional[str] = None,
                       user_mode: str = "patient",
                       processing_time: float = 0.0,
                       metadata: Optional[Dict] = None) -> str:
        """
        Log user interaction for audit trail
        
        Args:
            user_query: User's input query
            response_data: System's response data
            session_id: Optional session identifier
            user_mode: Patient or clinician mode
            processing_time: Time taken to process query
            metadata: Additional metadata
            
        Returns:
            Unique interaction ID
        """
        # Generate interaction ID
        interaction_id = hashlib.md5(
            f"{datetime.now().isoformat()}_{user_query[:50]}".encode()
        ).hexdigest()[:12]
        
        # Create audit record
        audit_record = {
            "interaction_id": interaction_id,
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id or "anonymous",
            "user_mode": user_mode,
            "query_hash": hashlib.md5(user_query.encode()).hexdigest(),
            "query_length": len(user_query),
            "processing_time": processing_time,
            "response_confidence": response_data.get('response', {}).get('confidence_overall', 0),
            "emergency_detected": response_data.get('emergency', False),
            "sources_used": response_data.get('passages_used', 0),
            "metadata": metadata or {}
        }
        
        # Log to file
        self.logger.info(f"INTERACTION: {json.dumps(audit_record)}")
        
        # Store for session analysis
        self.response_history.append({
            "interaction_id": interaction_id,
            "timestamp": datetime.now(),
            "user_query": user_query,
            "response_data": response_data,
            "session_id": session_id,
            "user_mode": user_mode
        })
        
        # Update session stats
        self.session_stats[session_id or "anonymous"] += 1
        
        return interaction_id
    
    def analyze_safety_risks(self, 
                           user_query: str, 
                           response_data: Dict[str, Any],
                           session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze potential safety risks in interaction
        
        Args:
            user_query: User's query
            response_data: System response
            session_id: Session identifier
            
        Returns:
            Safety risk analysis
        """
        risks = {
            "risk_level": "low",
            "risk_factors": [],
            "recommendations": [],
            "requires_human_review": False
        }
        
        query_lower = user_query.lower()
        
        # Check for emergency patterns
        emergency_matches = []
        for category, patterns in self.emergency_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    emergency_matches.append((category, pattern))
        
        if emergency_matches:
            risks["risk_level"] = "critical"
            risks["risk_factors"].append(f"Emergency patterns detected: {emergency_matches}")
            risks["recommendations"].append("Immediate medical attention recommended")
            risks["requires_human_review"] = True
        
        # Check response confidence
        confidence = response_data.get('response', {}).get('confidence_overall', 0)
        if confidence < self.safety_config["min_confidence_threshold"]:
            risks["risk_factors"].append(f"Low confidence response: {confidence:.2f}")
            risks["recommendations"].append("Consider consulting healthcare professional")
            if risks["risk_level"] == "low":
                risks["risk_level"] = "medium"
        
        # Check session volume
        session_query_count = self.session_stats.get(session_id or "anonymous", 0)
        if session_query_count > self.safety_config["max_queries_per_session"]:
            risks["risk_factors"].append(f"High query volume in session: {session_query_count}")
            risks["recommendations"].append("Consider rate limiting or human intervention")
            risks["requires_human_review"] = True
        
        # Check for potential misuse patterns
        if self._detect_suspicious_patterns(user_query, session_id):
            risks["risk_level"] = "high" if risks["risk_level"] != "critical" else "critical"
            risks["risk_factors"].append("Suspicious usage pattern detected")
            risks["requires_human_review"] = True
        
        # Log safety analysis
        safety_record = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "query_hash": hashlib.md5(user_query.encode()).hexdigest(),
            "risk_analysis": risks
        }
        self.logger.warning(f"SAFETY_ANALYSIS: {json.dumps(safety_record)}")
        
        return risks
    
    def _detect_suspicious_patterns(self, query: str, session_id: Optional[str]) -> bool:
        """Detect potentially suspicious usage patterns"""
        suspicious_indicators = [
            "drug", "prescription", "dosage", "medication list",
            "diagnose me", "am i sick", "what do i have",
            "medical records", "test results"
        ]
        
        query_lower = query.lower()
        suspicious_count = sum(1 for indicator in suspicious_indicators if indicator in query_lower)
        
        return suspicious_count >= self.safety_config["suspicious_pattern_threshold"]
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get safety summary for a session"""
        session_interactions = [
            interaction for interaction in self.response_history 
            if interaction["session_id"] == session_id
        ]
        
        if not session_interactions:
            return {"error": "Session not found"}
        
        # Calculate metrics
        total_queries = len(session_interactions)
        emergency_count = sum(
            1 for interaction in session_interactions 
            if interaction["response_data"].get("emergency", False)
        )
        
        avg_confidence = np.mean([
            interaction["response_data"].get("response", {}).get("confidence_overall", 0)
            for interaction in session_interactions
        ])
        
        return {
            "session_id": session_id,
            "total_queries": total_queries,
            "emergency_detections": emergency_count,
            "average_confidence": float(avg_confidence),
            "session_start": session_interactions[0]["timestamp"].isoformat(),
            "session_end": session_interactions[-1]["timestamp"].isoformat(),
            "safety_status": "safe" if emergency_count == 0 and avg_confidence > 0.5 else "needs_review"
        }


class ResponseEvaluator:
    """
    Evaluate response quality and medical accuracy
    """
    
    def __init__(self):
        """Initialize response evaluator"""
        self.evaluation_metrics = {
            "relevance": 0.0,
            "completeness": 0.0,
            "accuracy": 0.0,
            "safety": 0.0,
            "clarity": 0.0
        }
        
        self.evaluation_history = []
    
    def evaluate_response(self, 
                         user_query: str,
                         response_data: Dict[str, Any],
                         retrieved_passages: List[Dict],
                         ground_truth: Optional[Dict] = None) -> Dict[str, float]:
        """
        Evaluate response quality across multiple dimensions
        
        Args:
            user_query: Original user query
            response_data: System response
            retrieved_passages: Retrieved knowledge passages
            ground_truth: Optional ground truth for accuracy evaluation
            
        Returns:
            Evaluation scores for different metrics
        """
        scores = {}
        
        # Relevance: How well does response match query
        scores["relevance"] = self._evaluate_relevance(user_query, response_data, retrieved_passages)
        
        # Completeness: How comprehensive is the response
        scores["completeness"] = self._evaluate_completeness(response_data)
        
        # Safety: How safe/appropriate is the response
        scores["safety"] = self._evaluate_safety(response_data)
        
        # Clarity: How clear and understandable is the response
        scores["clarity"] = self._evaluate_clarity(response_data)
        
        # Accuracy: If ground truth available
        if ground_truth:
            scores["accuracy"] = self._evaluate_accuracy(response_data, ground_truth)
        
        # Calculate overall score
        scores["overall"] = np.mean(list(scores.values()))
        
        # Store evaluation
        evaluation_record = {
            "timestamp": datetime.now().isoformat(),
            "query": user_query[:100],  # Truncated for storage
            "scores": scores,
            "has_ground_truth": ground_truth is not None
        }
        self.evaluation_history.append(evaluation_record)
        
        return scores
    
    def _evaluate_relevance(self, query: str, response_data: Dict, passages: List[Dict]) -> float:
        """Evaluate response relevance to query"""
        if not response_data or response_data.get("error"):
            return 0.0
        
        # Simple heuristics for relevance
        response_content = response_data.get("response", {})
        
        # Check if response has content
        has_assessment = bool(response_content.get("summary"))
        has_conditions = bool(response_content.get("conditions"))
        has_recommendations = bool(response_content.get("recommended_actions"))
        
        content_score = (has_assessment + has_conditions + has_recommendations) / 3
        
        # Check confidence score
        confidence = response_content.get("confidence_overall", 0)
        
        # Check source utilization
        sources_used = response_data.get("passages_used", 0)
        source_score = min(sources_used / 3, 1.0)  # Normalize to 1.0 max
        
        return (content_score * 0.5 + confidence * 0.3 + source_score * 0.2)
    
    def _evaluate_completeness(self, response_data: Dict) -> float:
        """Evaluate response completeness"""
        if not response_data or response_data.get("error"):
            return 0.0
        
        response_content = response_data.get("response", {})
        
        # Check for different response components
        components = [
            "summary",
            "conditions", 
            "recommended_actions",
            "disclaimer"
        ]
        
        present_components = sum(1 for comp in components if response_content.get(comp))
        return present_components / len(components)
    
    def _evaluate_safety(self, response_data: Dict) -> float:
        """Evaluate response safety"""
        safety_score = 1.0
        
        # Emergency detection working properly
        if response_data.get("emergency"):
            safety_score = 1.0  # Good - emergency detected
        
        # Check for appropriate disclaimers
        response_content = response_data.get("response", {})
        if response_content.get("disclaimer"):
            safety_score = min(safety_score + 0.1, 1.0)
        
        # Check confidence appropriate for content
        confidence = response_content.get("confidence_overall", 0)
        if confidence > 0.8:  # Very high confidence might be unsafe for medical AI
            safety_score -= 0.2
        
        return max(safety_score, 0.0)
    
    def _evaluate_clarity(self, response_data: Dict) -> float:
        """Evaluate response clarity"""
        if not response_data or response_data.get("error"):
            return 0.0
        
        response_content = response_data.get("response", {})
        
        # Simple heuristics for clarity
        has_structured_format = bool(response_content.get("conditions") or response_content.get("recommended_actions"))
        has_summary = bool(response_content.get("summary"))
        
        # Assume good clarity if structured properly
        return (has_structured_format + has_summary) / 2
    
    def _evaluate_accuracy(self, response_data: Dict, ground_truth: Dict) -> float:
        """Evaluate response accuracy against ground truth"""
        # This would require labeled test data
        # For now, return a placeholder
        return 0.5
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations"""
        if not self.evaluation_history:
            return {"message": "No evaluations performed yet"}
        
        # Calculate averages
        all_scores = [eval_record["scores"] for eval_record in self.evaluation_history]
        
        avg_scores = {}
        for metric in ["relevance", "completeness", "safety", "clarity", "overall"]:
            scores = [scores.get(metric, 0) for scores in all_scores]
            avg_scores[f"avg_{metric}"] = float(np.mean(scores))
            avg_scores[f"std_{metric}"] = float(np.std(scores))
        
        return {
            "total_evaluations": len(self.evaluation_history),
            "average_scores": avg_scores,
            "last_evaluation": self.evaluation_history[-1]["timestamp"]
        }


class ComprehensiveSafetySystem:
    """
    Comprehensive safety and evaluation system
    """
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize comprehensive safety system"""
        self.safety_monitor = SafetyMonitor(log_dir)
        self.response_evaluator = ResponseEvaluator()
        
        print("🛡️ Comprehensive safety system initialized")
    
    def process_interaction(self,
                          user_query: str,
                          response_data: Dict[str, Any],
                          retrieved_passages: List[Dict],
                          session_id: Optional[str] = None,
                          user_mode: str = "patient",
                          processing_time: float = 0.0) -> Dict[str, Any]:
        """
        Process interaction through full safety and evaluation pipeline
        
        Args:
            user_query: User's query
            response_data: System response
            retrieved_passages: Retrieved passages
            session_id: Session ID
            user_mode: User mode (patient/clinician)
            processing_time: Processing time
            
        Returns:
            Comprehensive analysis results
        """
        # Log interaction
        interaction_id = self.safety_monitor.log_interaction(
            user_query=user_query,
            response_data=response_data,
            session_id=session_id,
            user_mode=user_mode,
            processing_time=processing_time
        )
        
        # Analyze safety risks
        safety_analysis = self.safety_monitor.analyze_safety_risks(
            user_query=user_query,
            response_data=response_data,
            session_id=session_id
        )
        
        # Evaluate response quality
        evaluation_scores = self.response_evaluator.evaluate_response(
            user_query=user_query,
            response_data=response_data,
            retrieved_passages=retrieved_passages
        )
        
        return {
            "interaction_id": interaction_id,
            "safety_analysis": safety_analysis,
            "evaluation_scores": evaluation_scores,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health and safety status"""
        eval_summary = self.response_evaluator.get_evaluation_summary()
        
        return {
            "safety_monitor_status": "active",
            "total_interactions": len(self.safety_monitor.response_history),
            "evaluation_summary": eval_summary,
            "system_status": "healthy",
            "last_interaction": self.safety_monitor.response_history[-1]["timestamp"].isoformat() if self.safety_monitor.response_history else None
        }


# Example usage and testing
if __name__ == "__main__":
    print("Healthcare Super-Assistant - Safety & Evaluation System Test")
    print("=" * 65)
    
    # Initialize safety system
    safety_system = ComprehensiveSafetySystem()
    
    # Test with sample interactions
    test_interactions = [
        {
            "query": "I have a headache and fever",
            "response": {
                "emergency": False,
                "response": {
                    "summary": "Possible viral infection symptoms",
                    "conditions": [{"name": "Common cold", "confidence": 0.7}],
                    "recommended_actions": ["Rest and hydration"],
                    "confidence_overall": 0.6,
                    "disclaimer": "Consult healthcare provider"
                },
                "passages_used": 3
            }
        },
        {
            "query": "I'm having severe chest pain and difficulty breathing",
            "response": {
                "emergency": True,
                "detected_keywords": ["chest pain", "difficulty breathing"],
                "passages_used": 2
            }
        }
    ]
    
    # Process test interactions
    for i, interaction in enumerate(test_interactions, 1):
        print(f"\nTest Interaction {i}:")
        print(f"Query: '{interaction['query']}'")
        
        result = safety_system.process_interaction(
            user_query=interaction["query"],
            response_data=interaction["response"],
            retrieved_passages=[],  # Mock empty for test
            session_id="test_session_1",
            processing_time=1.5
        )
        
        print(f"Safety Risk Level: {result['safety_analysis']['risk_level']}")
        print(f"Evaluation Overall Score: {result['evaluation_scores']['overall']:.2f}")
        print(f"Requires Review: {result['safety_analysis']['requires_human_review']}")
    
    # Get system health
    health = safety_system.get_system_health()
    print(f"\nSystem Health: {health['system_status']}")
    print(f"Total Interactions: {health['total_interactions']}")
    
    print("\n✅ Safety and evaluation system testing completed!")
