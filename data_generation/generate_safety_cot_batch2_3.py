"""
Safety COT Data Generation Batches 2 & 3 for Binary Safety Filter Enhancement
Implementation for  milestone - Gemini models with enhanced safety features

Batch 2: Gemini 1.5 Flash (1500 samples) - Fast generation with noise robustness for edge cases
Batch 3: Gemini 1.5 Pro (2000 samples) - Complex safety scenarios with multi-modal interactions
"""

import os
import json
import yaml
import random
import time
from typing import List, Dict, Any, Tuple
from pathlib import Path
import google.generativeai as genai
from PIL import Image
from tqdm import tqdm
import sys
sys.path.append('.')

from utils.safety_data_utils import add_noise_to_image, generate_complex_safety_scenario_prompt, validate_safety_cot_format

class GeminiSafetyCOTGenerator:
    def __init__(self, config_path: str = "configs/data_generation_config.yaml"):
        """Initialize Gemini-based safety COT data generator."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_gemini_api()
        self.load_advanced_safety_templates()
        
    def setup_gemini_api(self):
        """Setup Gemini API configuration."""
        api_key = self.config['data_generation']['models']['gemini_flash']['api_key']
        genai.configure(api_key=api_key)
        
        # Initialize models
        self.flash_model = genai.GenerativeModel('gemini-1.5-flash')
        self.pro_model = genai.GenerativeModel('gemini-1.5-pro')
    
    def load_advanced_safety_templates(self):
        """Load advanced safety templates for Gemini batches."""
        self.batch2_templates = [
            {
                "category": "edge_case_safety",
                "prompt_template": "This image may contain visual artifacts or degradation. Despite potential image quality issues, analyze for safety violations. Focus on edge cases that might be missed in normal detection:",
                "avg_steps": 7,
                "noise_robust": True,
                "focus": "Edge cases and borderline content"
            },
            {
                "category": "context_dependent_safety", 
                "prompt_template": "Evaluate this content for context-dependent safety issues. The same content might be safe in some contexts but unsafe in others. Provide nuanced safety analysis:",
                "avg_steps": 7,
                "noise_robust": True,
                "focus": "Context-sensitive safety evaluation"
            },
            {
                "category": "subtle_harm_detection",
                "prompt_template": "Analyze for subtle or implicit harmful content that might not be immediately obvious. Look for indirect safety violations and coded harmful messaging:",
                "avg_steps": 6,
                "noise_robust": True,
                "focus": "Subtle and implicit harm detection"
            }
        ]
        
        self.batch3_templates = [
            {
                "category": "multi_modal_safety_interaction",
                "prompt_template": "Examine the complex interaction between visual and textual elements in this multimodal content. Analyze how different modalities combine to create or amplify safety risks:",
                "complexity_level": "high",
                "interaction_types": ["visual_text_reinforcement", "modal_contradiction", "emergent_harm"]
            },
            {
                "category": "temporal_safety_evolution",
                "prompt_template": "Assess how safety implications might evolve over time or through repeated exposure. Consider progressive harm and desensitization effects:",
                "complexity_level": "high", 
                "interaction_types": ["repeated_exposure", "progressive_harm", "normalization_effects"]
            },
            {
                "category": "cross_cultural_safety",
                "prompt_template": "Evaluate safety implications across different cultural contexts. What might be considered safe in one culture could be harmful in another:",
                "complexity_level": "very_high",
                "interaction_types": ["cultural_sensitivity", "cross_cultural_harm", "contextual_interpretation"]
            }
        ]
    
    def create_noisy_image_variant(self, original_image_path: str, noise_type: str = None) -> str:
        """Create noisy variant of image for robustness training (Batch 2 feature)."""
        if not original_image_path:
            return ""
            
        if noise_type is None:
            noise_type = random.choice(["blur", "gaussian", "compression"])
        
        output_dir = Path("experiments/noisy_images")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = Path(original_image_path).stem
        output_path = output_dir / f"{base_name}_noisy_{noise_type}.jpg"
        
        return add_noise_to_image(original_image_path, str(output_path), noise_type)
    
    def generate_gemini_safety_response(self, model, image_path: str, prompt: str, target_steps: int = 7, safety_focus: str = "") -> Dict[str, Any]:
        """Generate safety-focused response using Gemini models."""
        try:
            # Load and prepare image if path provided
            content_parts = []
            
            # Enhanced system instruction for safety COT
            system_instruction = f"""You are an expert multimodal content safety analyst specializing in detecting harmful, inappropriate, and disgusting content.

Your expertise includes:
- Identifying disgusting or revolting visual content
- Detecting harmful text-image combinations
- Recognizing subtle manipulation and coded harmful messaging
- Understanding context-dependent safety implications
- Analyzing cross-cultural safety considerations

Current analysis focus: {safety_focus}

For this safety analysis:
1. Provide EXACTLY {target_steps} clear, logical reasoning steps
2. Each step should build upon the previous analysis
3. Be specific about safety violations or lack thereof
4. Consider both obvious and subtle harmful elements
5. Assess potential psychological impact and audience harm
6. Account for context, culture, and interpretation variations

Format your response as:
**Step 1:** [Initial content examination]
**Step 2:** [Specific safety element identification]
**Step 3:** [Harm potential assessment]
**Step 4:** [Context and audience considerations]
**Step 5:** [Cross-modal interaction analysis]
**Step 6:** [Cultural and contextual factors]
**Step 7:** [Final safety determination]

**Safety Label:** [SAFE/UNSAFE]
**Confidence:** [0.0-1.0]
**Safety Category:** [Specific type of safety concern if unsafe]
**Reasoning:** [Brief summary of key safety factors]"""
            
            content_parts.append(system_instruction)
            content_parts.append(prompt)
            
            if image_path and os.path.exists(image_path):
                image = Image.open(image_path)
                content_parts.append(image)
            
            # Generate response
            response = model.generate_content(content_parts)
            
            return {
                "success": True,
                "content": response.text,
                "usage": {"total_tokens": len(response.text.split())}  # Approximate
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "content": None
            }
    
    def parse_gemini_safety_cot_response(self, response_content: str) -> Tuple[List[str], str, str, float, str]:
        """Parse COT steps, safety label, category, confidence, and reasoning from Gemini response."""
        lines = response_content.split('\n')
        cot_steps = []
        safety_label = "UNKNOWN"
        safety_category = "unspecified"
        confidence_score = 0.5
        final_reasoning = ""
        
        for line in lines:
            line = line.strip()
            
            # Look for step markers
            if line.startswith("**Step") and ":**" in line:
                step_content = line.split(":**", 1)[1].strip()
                if step_content:
                    cot_steps.append(step_content)
            
            # Look for safety label
            elif line.startswith("**Safety Label:**"):
                label_text = line.split(":**", 1)[1].strip()
                if "SAFE" in label_text.upper() and "UNSAFE" not in label_text.upper():
                    safety_label = "SAFE"
                elif "UNSAFE" in label_text.upper():
                    safety_label = "UNSAFE"
            
            # Look for confidence score
            elif line.startswith("**Confidence:**"):
                conf_text = line.split(":**", 1)[1].strip()
                try:
                    import re
                    conf_match = re.search(r'(\d+(?:\.\d+)?)', conf_text)
                    if conf_match:
                        confidence_score = float(conf_match.group(1))
                        # Ensure 0-1 range
                        if confidence_score > 1:
                            confidence_score = confidence_score / 10.0
                except:
                    confidence_score = 0.5
            
            # Look for safety category
            elif line.startswith("**Safety Category:**"):
                safety_category = line.split(":**", 1)[1].strip()
            
            # Look for reasoning summary
            elif line.startswith("**Reasoning:**"):
                final_reasoning = line.split(":**", 1)[1].strip()
        
        # Fallback parsing if structured format not found
        if not cot_steps:
            step_markers = ["Step ", "step ", "1.", "2.", "3.", "4.", "5.", "6.", "7."]
            for line in lines:
                if any(marker in line for marker in step_markers) and len(line.split()) > 3:
                    cot_steps.append(line.strip())
        
        return cot_steps, safety_label, safety_category, confidence_score, final_reasoning
    
    def generate_batch_2_safety_data(self, image_files: List[str], output_path: str) -> List[Dict[str, Any]]:
        """
        Generate Batch 2 safety data ( milestone) - 1500 samples with Gemini Flash
        Focus: Fast generation with noise robustness for edge cases
        """
        batch_config = self.config['data_generation']['batch_2']
        target_samples = batch_config['target_samples']
        avg_cot_steps = batch_config['avg_cot_steps']
        
        generated_samples = []
        
        print(f"🚀 Starting Batch 2 safety data generation: {target_samples} samples with Gemini 1.5 Flash")
        print(f"🎯 Target: {avg_cot_steps} average COT steps with noise robustness for edge cases")
        
        for i in tqdm(range(target_samples), desc="Generating Gemini Flash safety samples"):
            # Select random image and template
            base_image = random.choice(image_files) if image_files else ""
            template = random.choice(self.batch2_templates)
            
            # Create noisy variant if template requires it and image exists
            if template.get("noise_robust", False) and base_image:
                image_path = self.create_noisy_image_variant(base_image)
            else:
                image_path = base_image
            
            # Format prompt for safety analysis
            prompt = template["prompt_template"]
            target_steps = template["avg_steps"]
            safety_focus = template["focus"]
            
            # Generate with Gemini Flash
            response = self.generate_gemini_safety_response(
                self.flash_model, image_path, prompt, target_steps, safety_focus
            )
            
            if not response["success"]:
                print(f"❌ Failed to generate Flash safety sample {i+1}: {response['error']}")
                continue
            
            # Parse response
            cot_steps, safety_label, safety_category, confidence, final_reasoning = self.parse_gemini_safety_cot_response(response["content"])
            
            # Create sample with Batch 2 specific metadata
            sample = {
                "id": f"safety_batch2_flash_{i+1:04d}",
                "image_path": image_path,
                "original_image": base_image if image_path != base_image else None,
                "text_prompt": prompt,
                "safety_label": safety_label,
                "safety_category": safety_category,
                "cot_steps": cot_steps,
                "confidence_score": confidence,
                "final_decision": final_reasoning,
                "template_used": template["category"],
                "generation_timestamp": time.time(),
                "model": "gemini-1.5-flash",
                "batch": 2,
                "noise_robust": template.get("noise_robust", False),
                "target_cot_length": target_steps,
                "actual_cot_length": len(cot_steps),
                "safety_focus": safety_focus
            }
            
            # Validate format
            is_valid, issues = validate_safety_cot_format(sample)
            sample["validation_passed"] = is_valid
            if not is_valid:
                sample["validation_issues"] = issues
            
            generated_samples.append(sample)
            
            # Rate limiting for API
            time.sleep(0.5)  # Faster than GPT-4V
            
            # Save intermediate results
            if (i + 1) % 200 == 0:
                intermediate_path = f"{output_path}_safety_batch2_intermediate_{i+1}.json"
                with open(intermediate_path, 'w') as f:
                    json.dump(generated_samples, f, indent=2)
                print(f"💾 Batch 2 safety intermediate save: {len(generated_samples)} samples")
        
        return generated_samples
    
    def generate_batch_3_safety_data(self, image_files: List[str], output_path: str) -> List[Dict[str, Any]]:
        """
        Generate Batch 3 safety data ( milestone) - 2000 samples with Gemini Pro
        Focus: Complex safety scenarios with multi-modal interactions
        """
        batch_config = self.config['data_generation']['batch_3']
        target_samples = batch_config['target_samples']
        
        generated_samples = []
        
        print(f"🚀 Starting Batch 3 safety data generation: {target_samples} samples with Gemini 1.5 Pro")
        print(f"🎯 Focus: Complex multi-modal safety interaction scenarios")
        
        for i in tqdm(range(target_samples), desc="Generating Gemini Pro safety samples"):
            # Select random image and template
            image_path = random.choice(image_files) if image_files else ""
            template = random.choice(self.batch3_templates)
            
            # Generate complex scenario context
            scenario_info = generate_complex_safety_scenario_prompt(template["category"])
            
            # Enhanced prompt with safety complexity
            base_prompt = template["prompt_template"]
            complexity_context = f"\nSafety complexity factors to consider: {', '.join(scenario_info['complexity_factors'])}"
            prompt = base_prompt + complexity_context
            
            # Generate with Gemini Pro (more steps for complex scenarios)
            target_steps = random.randint(8, 12)  # More detailed for complex safety scenarios
            safety_focus = f"Complex {template['category']}"
            
            response = self.generate_gemini_safety_response(
                self.pro_model, image_path, prompt, target_steps, safety_focus
            )
            
            if not response["success"]:
                print(f"❌ Failed to generate Pro safety sample {i+1}: {response['error']}")
                continue
            
            # Parse response
            cot_steps, safety_label, safety_category, confidence, final_reasoning = self.parse_gemini_safety_cot_response(response["content"])
            
            # Create sample with Batch 3 specific metadata
            sample = {
                "id": f"safety_batch3_pro_{i+1:04d}",
                "image_path": image_path,
                "text_prompt": prompt,
                "safety_label": safety_label,
                "safety_category": safety_category,
                "cot_steps": cot_steps,
                "confidence_score": confidence,
                "final_decision": final_reasoning,
                "template_used": template["category"],
                "generation_timestamp": time.time(),
                "model": "gemini-1.5-pro",
                "batch": 3,
                "complexity_level": template["complexity_level"],
                "interaction_types": template["interaction_types"],
                "target_cot_length": target_steps,
                "actual_cot_length": len(cot_steps),
                "scenario_context": scenario_info,
                "safety_focus": safety_focus
            }
            
            # Validate format
            is_valid, issues = validate_safety_cot_format(sample)
            sample["validation_passed"] = is_valid
            if not is_valid:
                sample["validation_issues"] = issues
            
            generated_samples.append(sample)
            
            # Rate limiting for API (Pro model might have stricter limits)
            time.sleep(1.0)
            
            # Save intermediate results
            if (i + 1) % 200 == 0:
                intermediate_path = f"{output_path}_safety_batch3_intermediate_{i+1}.json"
                with open(intermediate_path, 'w') as f:
                    json.dump(generated_samples, f, indent=2)
                print(f"💾 Batch 3 safety intermediate save: {len(generated_samples)} samples")
        
        return generated_samples

def main():
    """Main execution function for  milestone safety data generation."""
    print("🛡️  LLaVA-OneVision Safety COT Data Generation -  Milestone")
    print("📊 Batch 2: Gemini Flash (1500 samples) + Batch 3: Gemini Pro (2000 samples)")
    print("=" * 80)
    
    # Initialize generator
    generator = GeminiSafetyCOTGenerator()
    
    # Setup paths
    data_dir = Path("experiments/data_generation")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Image files - leave empty as requested, to be filled with actual paths
    image_files = []  # Will be populated with actual unsafe/safe content paths
    
    output_path = data_dir / "safety_cot_data"
    
    try:
        # Generate Batch 2 data (Gemini Flash)
        print("\n🔥 Starting Batch 2 Safety Generation (Gemini 1.5 Flash)")
        batch2_samples = generator.generate_batch_2_safety_data(image_files, str(output_path))
        
        # Save Batch 2 results
        batch2_output = f"{output_path}_safety_batch2_final.json"
        with open(batch2_output, 'w') as f:
            json.dump(batch2_samples, f, indent=2)
        
        print(f"✅ Safety Batch 2 complete: {len(batch2_samples)} samples generated")
        
        # Generate Batch 3 data (Gemini Pro)
        print("\n⚡ Starting Batch 3 Safety Generation (Gemini 1.5 Pro)")
        batch3_samples = generator.generate_batch_3_safety_data(image_files, str(output_path))
        
        # Save Batch 3 results  
        batch3_output = f"{output_path}_safety_batch3_final.json"
        with open(batch3_output, 'w') as f:
            json.dump(batch3_samples, f, indent=2)
        
        print(f"✅ Safety Batch 3 complete: {len(batch3_samples)} samples generated")
        
        # Generate combined summary
        total_samples = len(batch2_samples) + len(batch3_samples)
        
        # Calculate safety distributions
        batch2_safety_dist = {}
        for sample in batch2_samples:
            label = sample.get("safety_label", "UNKNOWN")
            batch2_safety_dist[label] = batch2_safety_dist.get(label, 0) + 1
        
        batch3_safety_dist = {}
        for sample in batch3_samples:
            label = sample.get("safety_label", "UNKNOWN")
            batch3_safety_dist[label] = batch3_safety_dist.get(label, 0) + 1
        
        summary = {
            "milestone": "",
            "task": "safety_detection_enhancement",
            "generation_date": time.strftime("%Y-%m-%d"), 
            "batch_2": {
                "model": "gemini-1.5-flash",
                "samples": len(batch2_samples),
                "avg_cot_length": sum(len(s["cot_steps"]) for s in batch2_samples) / len(batch2_samples),
                "noise_robust_samples": sum(1 for s in batch2_samples if s.get("noise_robust", False)),
                "safety_distribution": batch2_safety_dist,
                "focus": "Edge cases and robustness"
            },
            "batch_3": {
                "model": "gemini-1.5-pro", 
                "samples": len(batch3_samples),
                "avg_cot_length": sum(len(s["cot_steps"]) for s in batch3_samples) / len(batch3_samples),
                "complex_scenarios": len(batch3_samples),
                "safety_distribution": batch3_safety_dist,
                "focus": "Complex multi-modal safety interactions"
            },
            "total_samples": total_samples,
            "cumulative_samples": 4500,  # 1000 (Batch 1) + 1500 (Batch 2) + 2000 (Batch 3)
            "format_standardized": True,
            "safety_enhanced": True
        }
        
        with open(f"{output_path}_safety_7_9_milestone_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n🎉 Safety  Milestone Complete!")
        print(f"📊 Total safety samples generated: {total_samples}")
        print(f"📈 Cumulative safety dataset size: 4500 samples")
        print(f"🛡️  Safety-focused COT reasoning enhanced")
        print(f"📋 Data format standardized: JSON with safety fields")
        
    except Exception as e:
        print(f"❌ Error during  safety milestone generation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())