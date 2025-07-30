"""
DPO Data Preparation for Safety Enhancement
Implementation for  milestone - DPO data annotation and preparation

This script prepares preference data for Direct Preference Optimization (DPO)
to enhance safety reasoning in LLaVA-OneVision models.
"""

import os
import json
import yaml
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append('.')

from utils.safety_data_utils import load_safety_json_data, validate_safety_cot_format

@dataclass
class DPODataConfig:
    """Configuration for DPO data preparation."""
    pretrained_model_path: str = "experiments/training/sft_7b_safety_think_token"
    source_data_path: str = "experiments/data_generation"
    output_dir: str = "experiments/dpo"
    target_pairs: int = 2000
    negative_generation_methods: List[str] = None
    quality_threshold: float = 0.7
    
    def __post_init__(self):
        if self.negative_generation_methods is None:
            self.negative_generation_methods = [
                "circular_reasoning", "incorrect_safety_assessment", 
                "incomplete_analysis", "overconfident_wrong", "logic_gaps"
            ]

class SafetyDPODataPreparer:
    """Prepares preference pairs for safety-focused DPO training."""
    
    def __init__(self, config: DPODataConfig):
        self.config = config
        self.setup_output_dir()
        self.load_source_data()
        
    def setup_output_dir(self):
        """Setup output directory for DPO data."""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
    def load_source_data(self) -> List[Dict[str, Any]]:
        """Load source COT data for generating preference pairs."""
        print("📥 Loading source COT data for DPO preparation...")
        
        # Load previously generated safety COT data
        source_files = [
            f"{self.config.source_data_path}/safety_cot_data_safety_batch1_final.json",
            f"{self.config.source_data_path}/safety_cot_data_safety_batch2_final.json",
            f"{self.config.source_data_path}/safety_cot_data_safety_batch3_final.json"
        ]
        
        all_data = []
        for file_path in source_files:
            if os.path.exists(file_path):
                batch_data = load_safety_json_data(file_path)
                all_data.extend(batch_data)
                print(f"   Loaded {len(batch_data)} samples from {os.path.basename(file_path)}")
        
        print(f"📋 Total source samples loaded: {len(all_data)}")
        
        # Filter high-quality samples for DPO
        quality_samples = [
            sample for sample in all_data 
            if (sample.get("confidence_score", 0) >= self.config.quality_threshold and
                sample.get("validation_passed", False))
        ]
        
        print(f"🎯 High-quality samples for DPO: {len(quality_samples)}")
        self.source_data = quality_samples
        return quality_samples
    
    def generate_negative_cot_reasoning(self, original_sample: Dict[str, Any], 
                                       method: str) -> Dict[str, Any]:
        """
        Generate negative (rejected) COT reasoning examples.
        
        Methods:
        - circular_reasoning: Create circular logic in safety assessment
        - incorrect_safety_assessment: Wrong safety conclusion
        - incomplete_analysis: Missing critical safety considerations
        - overconfident_wrong: High confidence with wrong assessment
        - logic_gaps: Logical inconsistencies in reasoning chain
        """
        
        original_cot = original_sample.get("cot_steps", [])
        original_safety_label = original_sample.get("safety_label", "UNKNOWN")
        
        if method == "circular_reasoning":
            # Create circular reasoning that doesn't advance the analysis
            negative_cot = [
                "Step 1: This content needs safety evaluation to determine if it's safe",
                "Step 2: Looking at safety factors to assess safety level", 
                "Step 3: The safety depends on whether the content is safe or not",
                "Step 4: Based on safety considerations, evaluating safety",
                "Step 5: Therefore, the content safety is determined by its safety level"
            ]
            negative_label = original_safety_label  # Same conclusion but poor reasoning
            
        elif method == "incorrect_safety_assessment":
            # Flip the safety assessment
            negative_cot = original_cot.copy()
            negative_label = "UNSAFE" if original_safety_label == "SAFE" else "SAFE"
            
        elif method == "incomplete_analysis":
            # Take only first 2-3 steps, missing critical analysis
            negative_cot = original_cot[:min(3, len(original_cot))]
            if len(negative_cot) < 3:
                negative_cot.extend([
                    "Step: Partial analysis shows some concerning elements",
                    "Step: Need to consider more factors but skipping detailed review"
                ])
            negative_label = original_safety_label
            
        elif method == "overconfident_wrong":
            # Wrong assessment with overconfident language
            wrong_label = "UNSAFE" if original_safety_label == "SAFE" else "SAFE"
            negative_cot = [
                f"Step 1: Immediately clear this content is definitely {wrong_label.lower()}",
                "Step 2: No doubt whatsoever about this assessment",
                "Step 3: Completely obvious without need for detailed analysis",
                "Step 4: 100% certain of this safety determination",
                "Step 5: Final verdict is crystal clear and indisputable"
            ]
            negative_label = wrong_label
            
        elif method == "logic_gaps":
            # Logical inconsistencies and non-sequiturs
            negative_cot = [
                "Step 1: Content appears to contain visual elements",
                "Step 2: Safety can be determined by examining color schemes",
                "Step 3: Since Tuesday is a weekday, the content safety is affected",
                "Step 4: Popular content is usually safer than unpopular content",
                "Step 5: Therefore the safety assessment is conclusive"
            ]
            negative_label = random.choice(["SAFE", "UNSAFE"])
            
        else:
            # Default fallback method
            negative_cot = original_cot[:2] + ["Step: Analysis incomplete due to unknown factors"]
            negative_label = "UNKNOWN"
        
        # Create negative sample
        negative_sample = original_sample.copy()
        negative_sample.update({
            "cot_steps": negative_cot,
            "safety_label": negative_label,
            "confidence_score": max(0.1, original_sample.get("confidence_score", 0.5) - 0.3),
            "generation_method": method,
            "is_negative_example": True,
            "final_decision": f"Safety Label: {negative_label} (Generated via {method})"
        })
        
        return negative_sample
    
    def create_preference_pair(self, chosen_sample: Dict[str, Any], 
                              rejected_sample: Dict[str, Any]) -> Dict[str, Any]:
        """Create a preference pair for DPO training."""
        
        # Format the instruction (prompt)
        image_path = chosen_sample.get("image_path", "")
        text_prompt = chosen_sample.get("text_prompt", "")
        
        instruction = f"Analyze this multimodal content for safety violations. {text_prompt}"
        
        # Format chosen response (high-quality COT)
        chosen_think_tokens = "<think>" + "\n".join(chosen_sample.get("cot_steps", [])) + "</think>"
        chosen_response = f"{chosen_think_tokens} {chosen_sample.get('final_decision', '')}"
        
        # Format rejected response (low-quality COT)
        rejected_think_tokens = "<think>" + "\n".join(rejected_sample.get("cot_steps", [])) + "</think>"
        rejected_response = f"{rejected_think_tokens} {rejected_sample.get('final_decision', '')}"
        
        pair = {
            "instruction": instruction,
            "chosen": chosen_response,
            "rejected": rejected_response,
            "image_path": image_path,
            "chosen_safety_label": chosen_sample.get("safety_label", "UNKNOWN"),
            "rejected_safety_label": rejected_sample.get("safety_label", "UNKNOWN"),
            "chosen_confidence": chosen_sample.get("confidence_score", 0.5),
            "rejected_confidence": rejected_sample.get("confidence_score", 0.5),
            "rejection_method": rejected_sample.get("generation_method", "unknown"),
            "source_sample_id": chosen_sample.get("id", "unknown"),
            "pair_quality_score": self.calculate_pair_quality(chosen_sample, rejected_sample)
        }
        
        return pair
    
    def calculate_pair_quality(self, chosen: Dict[str, Any], rejected: Dict[str, Any]) -> float:
        """Calculate quality score for preference pair."""
        score = 0.0
        
        # Higher score if chosen has higher confidence
        if chosen.get("confidence_score", 0) > rejected.get("confidence_score", 0):
            score += 0.3
        
        # Higher score if chosen has more detailed COT
        chosen_cot_len = len(chosen.get("cot_steps", []))
        rejected_cot_len = len(rejected.get("cot_steps", []))
        if chosen_cot_len > rejected_cot_len:
            score += 0.2
        
        # Higher score if safety labels are consistent with quality
        if chosen.get("safety_label") != "UNKNOWN":
            score += 0.2
        
        # Higher score for certain rejection methods
        rejection_method = rejected.get("generation_method", "")
        if rejection_method in ["circular_reasoning", "logic_gaps"]:
            score += 0.3  # Clear quality difference
        
        return min(1.0, score)
    
    def generate_dpo_dataset(self) -> List[Dict[str, Any]]:
        """Generate complete DPO dataset with preference pairs."""
        print(f"🚀 Generating {self.config.target_pairs} preference pairs for DPO...")
        
        preference_pairs = []
        
        # Select high-quality samples as "chosen" examples
        chosen_candidates = [
            sample for sample in self.source_data
            if (sample.get("confidence_score", 0) >= 0.8 and
                len(sample.get("cot_steps", [])) >= 5 and
                sample.get("validation_passed", False))
        ]
        
        print(f"📊 Chosen candidates: {len(chosen_candidates)}")
        
        for i in tqdm(range(self.config.target_pairs), desc="Creating preference pairs"):
            # Select random chosen example
            chosen_sample = random.choice(chosen_candidates)
            
            # Generate rejected example using random method
            rejection_method = random.choice(self.config.negative_generation_methods)
            rejected_sample = self.generate_negative_cot_reasoning(chosen_sample, rejection_method)
            
            # Create preference pair
            pair = self.create_preference_pair(chosen_sample, rejected_sample)
            pair["pair_id"] = f"dpo_pair_{i+1:04d}"
            
            preference_pairs.append(pair)
        
        # Sort by quality score and take best pairs
        preference_pairs.sort(key=lambda x: x["pair_quality_score"], reverse=True)
        
        print(f"✅ Generated {len(preference_pairs)} preference pairs")
        
        # Generate statistics
        self.generate_dpo_statistics(preference_pairs)
        
        return preference_pairs
    
    def generate_dpo_statistics(self, preference_pairs: List[Dict[str, Any]]):
        """Generate statistics about the DPO dataset."""
        print("📊 Generating DPO dataset statistics...")
        
        # Method distribution
        method_counts = {}
        for pair in preference_pairs:
            method = pair.get("rejection_method", "unknown")
            method_counts[method] = method_counts.get(method, 0) + 1
        
        # Quality distribution
        quality_scores = [pair.get("pair_quality_score", 0) for pair in preference_pairs]
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        # Safety label distribution
        chosen_labels = {}
        rejected_labels = {}
        for pair in preference_pairs:
            chosen_label = pair.get("chosen_safety_label", "UNKNOWN")
            rejected_label = pair.get("rejected_safety_label", "UNKNOWN")
            
            chosen_labels[chosen_label] = chosen_labels.get(chosen_label, 0) + 1
            rejected_labels[rejected_label] = rejected_labels.get(rejected_label, 0) + 1
        
        stats = {
            "total_pairs": len(preference_pairs),
            "avg_pair_quality": avg_quality,
            "rejection_method_distribution": method_counts,
            "chosen_safety_distribution": chosen_labels,
            "rejected_safety_distribution": rejected_labels,
            "generation_date": "2024-07-31",
            "purpose": "DPO training for safety enhancement"
        }
        
        # Save statistics
        stats_file = Path(self.config.output_dir) / "dpo_dataset_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"   Average pair quality: {avg_quality:.2f}")
        print(f"   Rejection methods: {method_counts}")
        print(f"   Chosen safety labels: {chosen_labels}")
        print(f"💾 Statistics saved to {stats_file}")
    
    def prepare_dpo_training_format(self, preference_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert preference pairs to DPO training format."""
        print("🔄 Converting to DPO training format...")
        
        dpo_formatted = []
        
        for pair in preference_pairs:
            # DPO training format
            dpo_sample = {
                "instruction": pair["instruction"],
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
                "image": pair.get("image_path", ""),
                "metadata": {
                    "pair_id": pair.get("pair_id"),
                    "chosen_safety": pair.get("chosen_safety_label"),
                    "rejected_safety": pair.get("rejected_safety_label"),
                    "rejection_method": pair.get("rejection_method"),
                    "quality_score": pair.get("pair_quality_score")
                }
            }
            dpo_formatted.append(dpo_sample)
        
        return dpo_formatted
    
    def save_dpo_dataset(self, preference_pairs: List[Dict[str, Any]]):
        """Save DPO dataset in multiple formats."""
        output_dir = Path(self.config.output_dir)
        
        # Save raw preference pairs
        raw_file = output_dir / "safety_dpo_preference_pairs.json"
        with open(raw_file, 'w') as f:
            json.dump(preference_pairs, f, indent=2)
        
        # Save DPO training format
        dpo_formatted = self.prepare_dpo_training_format(preference_pairs)
        dpo_file = output_dir / "safety_dpo_training_data.json"
        with open(dpo_file, 'w') as f:
            json.dump(dpo_formatted, f, indent=2)
        
        # Save as CSV for analysis
        csv_data = []
        for pair in preference_pairs:
            csv_row = {
                "pair_id": pair.get("pair_id"),
                "chosen_safety_label": pair.get("chosen_safety_label"),
                "rejected_safety_label": pair.get("rejected_safety_label"),
                "rejection_method": pair.get("rejection_method"),
                "quality_score": pair.get("pair_quality_score"),
                "chosen_confidence": pair.get("chosen_confidence"),
                "rejected_confidence": pair.get("rejected_confidence")
            }
            csv_data.append(csv_row)
        
        csv_df = pd.DataFrame(csv_data)
        csv_file = output_dir / "safety_dpo_analysis.csv"
        csv_df.to_csv(csv_file, index=False)
        
        print(f"💾 DPO dataset saved:")
        print(f"   Raw pairs: {raw_file}")
        print(f"   Training format: {dpo_file}")
        print(f"   Analysis CSV: {csv_file}")

def main():
    """Main execution function for  milestone DPO preparation."""
    print("🎯 LLaVA-OneVision Safety DPO Data Preparation -  Milestone")
    print("=" * 70)
    
    # Setup configuration
    config = DPODataConfig()
    
    # Initialize DPO data preparer
    preparer = SafetyDPODataPreparer(config)
    
    try:
        # Generate DPO dataset
        preference_pairs = preparer.generate_dpo_dataset()
        
        # Save dataset
        preparer.save_dpo_dataset(preference_pairs)
        
        print(f"\n🎉  DPO Preparation Complete!")
        print(f"📊 Results summary:")
        print(f"   Preference pairs generated: {len(preference_pairs)}")
        print(f"   Ready for DPO training with safety focus")
        print(f"   Training parameters: 2 epochs, batch size 8, beta=0.1")
        print(f"   Next phase: DPO training and performance monitoring")
        
        return 0
        
    except Exception as e:
        print(f"❌ DPO preparation failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())