"""
Pre-training with Safety <think> Token Integration
Implementation for  milestone - Safety COT data as <think> token sequences

This script implements both SFT and warm-up training approaches for integrating
Chain-of-Thought safety reasoning data into LLaVA-OneVision using special <think> tokens.
"""

import os
import json
import yaml
import torch
import wandb
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq,
    HfArgumentParser
)
from datasets import Dataset
from tqdm import tqdm
import deepspeed
import sys
sys.path.append('.')

from utils.safety_data_utils import load_safety_json_data, validate_safety_cot_format

@dataclass
class SafetyThinkTokenTrainingArguments:
    """Custom training arguments for safety <think> token integration."""
    config_path: str = field(default="configs/training_config.yaml")
    training_mode: str = field(default="sft", metadata={"choices": ["sft", "warmup"]})
    data_path: str = field(default="experiments/data_generation")
    model_name_or_path: str = field(default="llava-hf/llava-onevision-qwen2-7b-ov")
    output_dir: str = field(default="experiments/training/sft_7b_safety_think_token")
    
    # Data selection parameters
    max_samples: int = field(default=4000)
    min_cot_length: int = field(default=5)
    max_cot_length: int = field(default=12)
    quality_filter: bool = field(default=True)
    min_confidence: float = field(default=0.6)
    balance_safety_labels: bool = field(default=True)

class SafetyThinkTokenDataProcessor:
    """Process safety COT data into <think> token format for training."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.think_start_token = config['training']['data']['think_token']['start_token']
        self.think_end_token = config['training']['data']['think_token']['end_token']
        
    def format_safety_cot_as_think_tokens(self, cot_steps: List[str]) -> str:
        """
        Convert safety COT steps into <think> token format.
        
        Example:
        Input: ["Step 1: Examine visual content", "Step 2: Assess safety violations", ...]
        Output: "<think>Step 1: Examine visual content\nStep 2: Assess safety violations\n...</think>"
        """
        if not cot_steps:
            return ""
        
        # Join COT steps with newlines
        safety_cot_content = "\n".join(cot_steps)
        
        # Wrap in think tokens
        think_sequence = f"{self.think_start_token}{safety_cot_content}{self.think_end_token}"
        
        return think_sequence
    
    def create_safety_training_sample(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """
        Create a safety training sample in instruction-following format with <think> tokens.
        
        Format:
        User: [Safety evaluation prompt]
        Assistant: <think>[Safety COT steps]</think> [Safety decision with reasoning]
        """
        # Extract components
        prompt = sample.get("text_prompt", "")
        cot_steps = sample.get("cot_steps", [])
        safety_label = sample.get("safety_label", "UNKNOWN")
        safety_category = sample.get("safety_category", "unspecified")
        final_decision = sample.get("final_decision", "")
        confidence_score = sample.get("confidence_score", 0.5)
        image_path = sample.get("image_path", "")
        
        # Format COT as think tokens
        think_sequence = self.format_safety_cot_as_think_tokens(cot_steps)
        
        # Create safety instruction format
        user_message = f"Analyze this multimodal content for safety violations. {prompt}"
        
        # Create comprehensive assistant response
        safety_verdict = f"Safety Label: {safety_label}"
        if safety_label == "UNSAFE":
            safety_verdict += f" (Category: {safety_category})"
        safety_verdict += f"\nConfidence: {confidence_score:.2f}"
        if final_decision:
            safety_verdict += f"\nReasoning: {final_decision}"
        
        assistant_message = f"{think_sequence} {safety_verdict}"
        
        return {
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message}
            ],
            "image_path": image_path,
            "original_sample_id": sample.get("id", "unknown"),
            "safety_label": safety_label,
            "safety_category": safety_category,
            "confidence_score": confidence_score
        }
    
    def select_high_quality_safety_samples(self, all_samples: List[Dict[str, Any]], 
                                          max_samples: int, min_cot: int, max_cot: int,
                                          min_confidence: float, balance_labels: bool) -> List[Dict[str, Any]]:
        """Select high-quality safety samples based on specified criteria."""
        
        # Filter samples based on quality criteria
        filtered_samples = []
        
        for sample in all_samples:
            # Validate format
            is_valid, _ = validate_safety_cot_format(sample)
            if not is_valid:
                continue
            
            # Check COT length
            cot_length = len(sample.get("cot_steps", []))
            if not (min_cot <= cot_length <= max_cot):
                continue
            
            # Check confidence threshold
            confidence = sample.get("confidence_score", 0.0)
            if confidence < min_confidence:
                continue
            
            # Check quality approval (if available)
            if sample.get("quality_approved", True):  # Default to True if not specified
                filtered_samples.append(sample)
        
        # Balance safe vs unsafe samples if requested
        if balance_labels:
            safe_samples = [s for s in filtered_samples if s.get("safety_label") == "SAFE"]
            unsafe_samples = [s for s in filtered_samples if s.get("safety_label") == "UNSAFE"]
            
            # Take equal numbers of safe and unsafe, up to max_samples
            target_per_label = max_samples // 2
            
            selected_safe = safe_samples[:target_per_label]
            selected_unsafe = unsafe_samples[:target_per_label]
            
            selected = selected_safe + selected_unsafe
            
            print(f"🔄 Balanced dataset: {len(selected_safe)} safe + {len(selected_unsafe)} unsafe = {len(selected)} total")
            
        else:
            # Sort by quality metrics and take top samples
            def quality_score(sample):
                score = 0
                
                # Higher score for quality approved samples
                if sample.get("quality_approved", False):
                    score += 10
                
                # Higher score for higher confidence
                confidence = sample.get("confidence_score", 0.0)
                score += confidence * 5
                
                # Moderate COT length gets higher score
                cot_len = len(sample.get("cot_steps", []))
                if 6 <= cot_len <= 8:
                    score += 5
                
                # Prefer newer batches
                batch = sample.get("batch", 1)
                score += batch
                
                # Prefer samples that passed validation
                if sample.get("validation_passed", False):
                    score += 3
                
                return score
            
            filtered_samples.sort(key=quality_score, reverse=True)
            selected = filtered_samples[:max_samples]
        
        print(f"📊 Safety sample selection complete:")
        print(f"   Total available: {len(all_samples)}")
        print(f"   After filtering: {len(filtered_samples)}")
        print(f"   Selected for training: {len(selected)}")
        
        # Print safety distribution
        safety_dist = {}
        for sample in selected:
            label = sample.get("safety_label", "UNKNOWN")
            safety_dist[label] = safety_dist.get(label, 0) + 1
        print(f"   Safety distribution: {safety_dist}")
        
        return selected

class SafetyThinkTokenTrainer:
    """Custom trainer for safety <think> token integration."""
    
    def __init__(self, args: SafetyThinkTokenTrainingArguments):
        self.args = args
        
        # Load configuration
        with open(args.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_processor = SafetyThinkTokenDataProcessor(self.config)
        
        # Initialize wandb if configured
        if self.config['training']['logging']['use_wandb']:
            wandb.init(
                project=self.config['training']['logging']['project_name'],
                name=self.config['training']['logging']['experiment_name'] + f"_{args.training_mode}",
                config=self.config
            )
    
    def load_and_process_safety_data(self) -> Dataset:
        """Load safety COT data and convert to <think> token format."""
        
        print("📥 Loading safety COT data for <think> token training...")
        
        # Load all generated safety data batches
        data_files = [
            f"{self.args.data_path}/safety_cot_data_safety_batch1_final.json",
            f"{self.args.data_path}/safety_cot_data_safety_batch2_final.json", 
            f"{self.args.data_path}/safety_cot_data_safety_batch3_final.json"
        ]
        
        all_samples = []
        for file_path in data_files:
            if os.path.exists(file_path):
                batch_data = load_safety_json_data(file_path)
                all_samples.extend(batch_data)
                print(f"   Loaded {len(batch_data)} safety samples from {os.path.basename(file_path)}")
        
        print(f"📋 Total safety samples loaded: {len(all_samples)}")
        
        # Select high-quality safety samples
        selected_samples = self.data_processor.select_high_quality_safety_samples(
            all_samples, 
            self.args.max_samples,
            self.args.min_cot_length,
            self.args.max_cot_length,
            self.args.min_confidence,
            self.args.balance_safety_labels
        )
        
        # Convert to training format
        training_data = []
        for sample in tqdm(selected_samples, desc="Processing safety samples"):
            try:
                training_sample = self.data_processor.create_safety_training_sample(sample)
                training_data.append(training_sample)
            except Exception as e:
                print(f"⚠️  Error processing safety sample {sample.get('id', 'unknown')}: {e}")
        
        print(f"✅ Processed {len(training_data)} safety samples for training")
        
        # Create Hugging Face dataset
        dataset = Dataset.from_list(training_data)
        
        return dataset
    
    def setup_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Setup model and tokenizer with safety <think> token support."""
        
        print(f"🤖 Loading model: {self.args.model_name_or_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name_or_path,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add special safety <think> tokens
        special_tokens = [
            self.data_processor.think_start_token,
            self.data_processor.think_end_token
        ]
        
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        print(f"🔤 Added {num_added} special safety tokens: {special_tokens}")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Resize embeddings to accommodate new tokens
        if num_added > 0:
            model.resize_token_embeddings(len(tokenizer))
            print(f"📏 Resized model embeddings to {len(tokenizer)} tokens")
        
        return model, tokenizer
    
    def run_sft_training(self):
        """Run Supervised Fine-Tuning with safety <think> tokens."""
        
        print("🚀 Starting Safety SFT Training with <think> tokens")
        print("=" * 60)
        
        # Load data and model
        dataset = self.load_and_process_safety_data()
        model, tokenizer = self.setup_model_and_tokenizer()
        
        # Get SFT configuration
        sft_config = self.config['training']['sft']
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            num_train_epochs=sft_config['epochs'],
            per_device_train_batch_size=sft_config['batch_size'],
            gradient_accumulation_steps=sft_config['gradient_accumulation_steps'],
            learning_rate=sft_config['learning_rate'],
            warmup_steps=sft_config['warmup_steps'],
            logging_steps=sft_config['logging_steps'],
            save_steps=sft_config['save_steps'],
            eval_steps=sft_config['eval_steps'],
            save_total_limit=sft_config['save_total_limit'],
            dataloader_num_workers=sft_config['dataloader_num_workers'],
            bf16=True,
            gradient_checkpointing=True,
            deepspeed=self.config['training']['hardware']['deepspeed_config'],
            report_to="wandb" if self.config['training']['logging']['use_wandb'] else "none"
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            return_tensors="pt"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        # Start training
        print("💪 Starting safety SFT training...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        tokenizer.save_pretrained(self.args.output_dir)
        
        print("✅ Safety SFT Training Complete!")
        
    def run_warmup_training(self):
        """Run warm-up training with gradual safety COT injection."""
        
        print("🔥 Starting Warm-up Training with gradual safety COT injection")
        print("=" * 60)
        
        # This is a simplified version - full implementation would require
        # custom training loop with dynamic data loading for safety samples
        
        print("⚠️  Warm-up training with gradual safety injection requires custom implementation")
        print("📋 Implementation plan for safety training:")
        print("   1. Start with 10% safety COT data at low learning rate")
        print("   2. Gradually increase safety COT ratio every 100-200 steps")
        print("   3. Monitor safety accuracy and adjust injection schedule")
        print("   4. Balance safe/unsafe samples throughout training")
        print("   5. Complete with 100% safety COT data")
        
        # For now, run regular SFT with warmup config
        self.run_sft_training()

def main():
    """Main training execution function for  milestone."""
    
    print("🛡️  LLaVA-OneVision Safety <think> Token Pre-training -  Milestone")
    print("=" * 70)
    
    # Parse arguments
    parser = HfArgumentParser(SafetyThinkTokenTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    print(f"🔧 Training mode: {args.training_mode}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📊 Max samples: {args.max_samples}")
    print(f"🎯 Min confidence: {args.min_confidence}")
    print(f"⚖️  Balance safety labels: {args.balance_safety_labels}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = SafetyThinkTokenTrainer(args)
    
    try:
        if args.training_mode == "sft":
            trainer.run_sft_training()
        elif args.training_mode == "warmup":
            trainer.run_warmup_training()
        else:
            raise ValueError(f"Unknown training mode: {args.training_mode}")
            
        print(f"\n🎉  Safety Milestone Training Complete!")
        print(f"📁 Safety-enhanced model saved to: {args.output_dir}")
        print(f"⏰ Expected training time: ~24 hours (single GPU)")
        print(f"🛡️  Model now enhanced with safety <think> token reasoning")
        
    except Exception as e:
        print(f"❌ Safety training failed: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())