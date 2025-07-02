"""
Utility functions for safety data processing and management
Supporting safety COT data generation and preprocessing tasks
"""

import json
import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
from PIL import Image, ImageFilter
import random

def load_safety_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load safety JSON data from file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_safety_json_data(data: List[Dict[str, Any]], file_path: str):
    """Save safety data to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def merge_safety_datasets(dataset_paths: List[str], output_path: str) -> int:
    """Merge multiple safety datasets into a single file."""
    merged_data = []
    
    for path in dataset_paths:
        if os.path.exists(path):
            data = load_safety_json_data(path)
            merged_data.extend(data)
    
    save_safety_json_data(merged_data, output_path)
    return len(merged_data)

def add_noise_to_image(image_path: str, output_path: str, noise_type: str = "blur") -> str:
    """
    Add noise elements to images for robustness training ( milestone feature).
    
    Args:
        image_path: Path to input image
        output_path: Path for output image
        noise_type: Type of noise ("blur", "gaussian", "compression")
    
    Returns:
        Path to the modified image
    """
    if not image_path or not os.path.exists(image_path):
        return ""
        
    image = Image.open(image_path)
    
    if noise_type == "blur":
        # Add blur for robustness testing
        blur_radius = random.uniform(0.5, 2.0)
        image = image.filter(ImageFilter.GaussianBlur(blur_radius))
    
    elif noise_type == "gaussian":
        # Add gaussian noise
        img_array = np.array(image)
        noise = np.random.normal(0, 15, img_array.shape).astype(np.uint8)
        img_array = np.clip(img_array + noise, 0, 255)
        image = Image.fromarray(img_array)
    
    elif noise_type == "compression":
        # Simulate compression artifacts
        # Save with low quality and reload
        temp_path = output_path.replace('.jpg', '_temp.jpg')
        image.save(temp_path, 'JPEG', quality=30)
        image = Image.open(temp_path)
        os.remove(temp_path)
    
    image.save(output_path)
    return output_path

def validate_safety_cot_format(sample: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate safety COT sample format and return validation status with issues.
    
    Expected format from  milestone:
    - image_path: str
    - text_prompt: str  
    - safety_label: str (SAFE/UNSAFE)
    - safety_category: str
    - cot_steps: List[str]
    - confidence_score: float
    - final_decision: str
    """
    issues = []
    
    # Check required fields
    required_fields = ["image_path", "text_prompt", "safety_label", "safety_category", "cot_steps", "confidence_score", "final_decision"]
    for field in required_fields:
        if field not in sample:
            issues.append(f"Missing required field: {field}")
    
    # Check data types
    if "cot_steps" in sample and not isinstance(sample["cot_steps"], list):
        issues.append("cot_steps must be a list")
    
    if "confidence_score" in sample and not isinstance(sample["confidence_score"], (int, float)):
        issues.append("confidence_score must be a number")
    
    # Check safety label validity
    if "safety_label" in sample and sample["safety_label"] not in ["SAFE", "UNSAFE"]:
        issues.append("safety_label must be either 'SAFE' or 'UNSAFE'")
    
    # Check reasonable COT length
    if "cot_steps" in sample and isinstance(sample["cot_steps"], list):
        if len(sample["cot_steps"]) < 3:
            issues.append("COT should have at least 3 steps")
        if len(sample["cot_steps"]) > 15:
            issues.append("COT should have at most 15 steps")
    
    # Check confidence score range
    if "confidence_score" in sample:
        score = sample["confidence_score"]
        if not (0 <= score <= 1):
            issues.append("confidence_score should be between 0 and 1")
    
    # Check if image file exists (if path provided)
    if "image_path" in sample and sample["image_path"] and not os.path.exists(sample["image_path"]):
        issues.append(f"Image file not found: {sample['image_path']}")
    
    return len(issues) == 0, issues

def generate_complex_safety_scenario_prompt(scenario_type: str) -> Dict[str, Any]:
    """
    Generate prompts for complex safety scenarios ( milestone - Gemini Pro batch).
    
    Args:
        scenario_type: Type of complex safety scenario
        
    Returns:
        Dictionary with prompt and context information
    """
    complex_safety_scenarios = {
        "multi_modal_manipulation": {
            "prompt": "Analyze this content for sophisticated manipulation combining visual and textual elements to create harmful messaging. Consider how seemingly innocent elements might combine to create unsafe content.",
            "complexity_factors": [
                "Subtle visual-text interactions",
                "Context-dependent harmful implications",
                "Cultural sensitivity considerations",
                "Plausible deniability in harmful content"
            ]
        },
        
        "temporal_safety_patterns": {
            "prompt": "Examine this content sequence for safety violations that emerge over time or through repeated exposure. Consider cumulative effects and pattern-based harm.",
            "complexity_factors": [
                "Repeated exposure effects",
                "Gradual normalization of harmful content",
                "Time-based safety degradation",
                "Sequential harm amplification"
            ]
        },
        
        "cross_modal_harm": {
            "prompt": "Evaluate how this multimodal content creates harm through the interaction between different modalities (visual, textual, contextual). Focus on emergent harmful properties.",
            "complexity_factors": [
                "Inter-modal reinforcement of harmful themes",
                "Cross-modal contradiction exploitation",
                "Emergent harmful semantics",
                "Modality-specific bypass attempts"
            ]
        },
        
        "contextual_safety_inversion": {
            "prompt": "Assess how this content's safety status might change dramatically based on context, audience, or interpretation. Analyze context-dependent safety risks.",
            "complexity_factors": [
                "Context-dependent interpretation",
                "Audience-specific harm potential",
                "Cultural context sensitivity",
                "Intent vs. impact analysis"
            ]
        }
    }
    
    return complex_safety_scenarios.get(scenario_type, complex_safety_scenarios["multi_modal_manipulation"])

def calculate_safety_dataset_statistics(dataset_path: str) -> Dict[str, Any]:
    """Calculate comprehensive statistics for a safety COT dataset."""
    data = load_safety_json_data(dataset_path)
    
    stats = {
        "total_samples": len(data),
        "safety_distribution": {},
        "category_distribution": {},
        "avg_cot_length": 0,
        "avg_confidence": 0,
        "min_cot_length": float('inf'),
        "max_cot_length": 0,
        "models_used": set(),
        "quality_metrics": {}
    }
    
    if not data:
        return stats
    
    cot_lengths = []
    confidence_scores = []
    
    for sample in data:
        # Safety label distribution
        if "safety_label" in sample:
            label = sample["safety_label"]
            stats["safety_distribution"][label] = stats["safety_distribution"].get(label, 0) + 1
        
        # Category distribution
        if "safety_category" in sample:
            category = sample["safety_category"]
            stats["category_distribution"][category] = stats["category_distribution"].get(category, 0) + 1
        
        # COT length analysis
        if "cot_steps" in sample and isinstance(sample["cot_steps"], list):
            cot_len = len(sample["cot_steps"])
            cot_lengths.append(cot_len)
            stats["min_cot_length"] = min(stats["min_cot_length"], cot_len)
            stats["max_cot_length"] = max(stats["max_cot_length"], cot_len)
        
        # Confidence score analysis
        if "confidence_score" in sample and isinstance(sample["confidence_score"], (int, float)):
            confidence_scores.append(sample["confidence_score"])
        
        # Model tracking
        if "model" in sample:
            stats["models_used"].add(sample["model"])
        
        # Quality metrics
        if "quality_approved" in sample:
            if "quality_approved" not in stats["quality_metrics"]:
                stats["quality_metrics"]["quality_approved"] = 0
            if sample["quality_approved"]:
                stats["quality_metrics"]["quality_approved"] += 1
    
    # Calculate averages
    if cot_lengths:
        stats["avg_cot_length"] = sum(cot_lengths) / len(cot_lengths)
    if confidence_scores:
        stats["avg_confidence"] = sum(confidence_scores) / len(confidence_scores)
    
    # Convert sets to lists for JSON serialization
    stats["models_used"] = list(stats["models_used"])
    
    # Quality approval rate
    if "quality_approved" in stats["quality_metrics"]:
        total_quality_checked = sum(1 for s in data if "quality_approved" in s)
        if total_quality_checked > 0:
            stats["quality_metrics"]["approval_rate"] = stats["quality_metrics"]["quality_approved"] / total_quality_checked
    
    # Safety detection accuracy metrics
    stats["safety_metrics"] = {
        "safe_samples": stats["safety_distribution"].get("SAFE", 0),
        "unsafe_samples": stats["safety_distribution"].get("UNSAFE", 0),
        "safety_ratio": stats["safety_distribution"].get("SAFE", 0) / len(data) if data else 0
    }
    
    return stats

def deduplicate_safety_dataset(dataset_path: str, output_path: str, similarity_threshold: float = 0.8) -> int:
    """
    Remove near-duplicate samples from safety dataset based on content similarity.
    
    Returns:
        Number of samples removed
    """
    data = load_safety_json_data(dataset_path)
    
    # Safety-focused deduplication
    unique_samples = []
    removed_count = 0
    
    for sample in data:
        is_duplicate = False
        current_prompt = sample.get("text_prompt", "")
        current_category = sample.get("safety_category", "")
        
        for existing_sample in unique_samples:
            existing_prompt = existing_sample.get("text_prompt", "")
            existing_category = existing_sample.get("safety_category", "")
            
            # Check category match first
            if current_category == existing_category:
                # Simple word overlap similarity within same category
                current_words = set(current_prompt.lower().split())
                existing_words = set(existing_prompt.lower().split())
                
                if current_words and existing_words:
                    overlap = len(current_words.intersection(existing_words))
                    union = len(current_words.union(existing_words))
                    similarity = overlap / union if union > 0 else 0
                    
                    if similarity > similarity_threshold:
                        is_duplicate = True
                        break
        
        if not is_duplicate:
            unique_samples.append(sample)
        else:
            removed_count += 1
    
    save_safety_json_data(unique_samples, output_path)
    return removed_count

def filter_safety_samples_by_confidence(dataset_path: str, output_path: str, min_confidence: float = 0.7) -> int:
    """
    Filter safety samples by confidence threshold.
    
    Returns:
        Number of samples that meet confidence threshold
    """
    data = load_safety_json_data(dataset_path)
    
    filtered_samples = []
    for sample in data:
        confidence = sample.get("confidence_score", 0.0)
        if confidence >= min_confidence:
            filtered_samples.append(sample)
    
    save_safety_json_data(filtered_samples, output_path)
    return len(filtered_samples)

def balance_safety_dataset(dataset_path: str, output_path: str, target_ratio: float = 0.5) -> Dict[str, int]:
    """
    Balance safety dataset to achieve target ratio of safe vs unsafe samples.
    
    Args:
        target_ratio: Target ratio of safe samples (0.5 = 50% safe, 50% unsafe)
    
    Returns:
        Dictionary with final counts
    """
    data = load_safety_json_data(dataset_path)
    
    safe_samples = [s for s in data if s.get("safety_label") == "SAFE"]
    unsafe_samples = [s for s in data if s.get("safety_label") == "UNSAFE"]
    
    total_samples = len(data)
    target_safe_count = int(total_samples * target_ratio)
    target_unsafe_count = total_samples - target_safe_count
    
    # Sample to achieve target distribution
    balanced_samples = []
    
    if len(safe_samples) >= target_safe_count:
        balanced_samples.extend(random.sample(safe_samples, target_safe_count))
    else:
        balanced_samples.extend(safe_samples)
    
    if len(unsafe_samples) >= target_unsafe_count:
        balanced_samples.extend(random.sample(unsafe_samples, target_unsafe_count))
    else:
        balanced_samples.extend(unsafe_samples)
    
    # Shuffle the balanced dataset
    random.shuffle(balanced_samples)
    
    save_safety_json_data(balanced_samples, output_path)
    
    return {
        "safe_samples": len([s for s in balanced_samples if s.get("safety_label") == "SAFE"]),
        "unsafe_samples": len([s for s in balanced_samples if s.get("safety_label") == "UNSAFE"]),
        "total_samples": len(balanced_samples)
    }