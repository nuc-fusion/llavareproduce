"""
Safety COT Data Generation for Binary Safety Filter Enhancement
This script generates Chain-of-Thought enhanced safety detection data
for identifying unsafe, disgusting, or harmful multimodal content.
"""

import os
import json
import yaml
import base64
import random
from typing import List, Dict, Any, Tuple
from pathlib import Path
import openai
from PIL import Image
import requests
from tqdm import tqdm
import time

class SafetyCOTDataGenerator:
    def __init__(self, config_path: str = "configs/data_generation_config.yaml"):
        """Initialize the safety COT data generator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_apis()
        self.safety_templates = self.load_safety_templates()
        
    def setup_apis(self):
        """Setup API configurations for different models."""
        openai.api_key = self.config['data_generation']['models']['gpt4v']['api_key']
        
    def load_safety_templates(self) -> List[Dict]:
        """Load safety detection templates for prompt generation."""
        return [
            {
                "category": "disgusting_visual",
                "prompt_template": "Analyze this image for disgusting or disturbing visual content. Evaluate step by step whether the image contains content that might be considered disgusting, revolting, or disturbing:",
                "cot_template": [
                    "Step 1: Examine the overall visual content and subject matter",
                    "Step 2: Identify any elements that could be considered disgusting or disturbing",
                    "Step 3: Assess the severity and nature of potentially harmful content",
                    "Step 4: Consider cultural and contextual factors in safety assessment",
                    "Step 5: Make final safety determination with clear reasoning"
                ],
                "safety_focus": "Visual content analysis for disgusting/disturbing material"
            },
            {
                "category": "harmful_text_image",
                "prompt_template": "Evaluate this image-text combination for harmful content. Analyze step by step how the text and image together might create unsafe or harmful messaging:",
                "cot_template": [
                    "Step 1: Analyze the text content for harmful language or implications",
                    "Step 2: Examine the image for concerning visual elements",
                    "Step 3: Assess how text and image interact to create meaning",
                    "Step 4: Identify potential harm from the combined message",
                    "Step 5: Determine overall safety level of the multimodal content"
                ],
                "safety_focus": "Multimodal harm detection in text-image combinations"
            },
            {
                "category": "inappropriate_content",
                "prompt_template": "Assess this content for appropriateness and safety. Use systematic reasoning to determine if this content violates safety guidelines:",
                "cot_template": [
                    "Step 1: Identify the main subjects and themes in the content",
                    "Step 2: Check for explicit, violent, or inappropriate material",
                    "Step 3: Evaluate context and potential audience impact",
                    "Step 4: Consider platform safety guidelines and policies",
                    "Step 5: Provide clear safety classification with justification"
                ],
                "safety_focus": "General content appropriateness and policy compliance"
            },
            {
                "category": "violence_graphic",
                "prompt_template": "Examine this content for graphic violence or disturbing imagery. Reason through the safety implications step by step:",
                "cot_template": [
                    "Step 1: Scan for violent actions, weapons, or aggressive behavior",
                    "Step 2: Assess the graphic nature and intensity of any violence",
                    "Step 3: Evaluate psychological impact and trauma potential",
                    "Step 4: Consider age-appropriateness and content warnings needed",
                    "Step 5: Classify safety level and recommend appropriate handling"
                ],
                "safety_focus": "Violence and graphic content detection"
            },
            {
                "category": "context_dependent_safety",
                "prompt_template": "Analyze this content considering contextual safety factors. Some content may be safe in certain contexts but unsafe in others. Reason through the contextual safety assessment:",
                "cot_template": [
                    "Step 1: Identify the apparent context and intended use",
                    "Step 2: Consider alternative interpretations or misuse potential",
                    "Step 3: Evaluate safety across different audience segments",
                    "Step 4: Assess potential for harm if taken out of context",
                    "Step 5: Provide nuanced safety assessment with context considerations"
                ],
                "safety_focus": "Context-aware safety evaluation"
            }
        ]
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API calls."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def generate_safety_prompt(self, template: Dict, additional_context: str = "") -> Tuple[str, List[str]]:
        """Generate safety detection prompt with COT steps."""
        # Format prompt template
        prompt = template["prompt_template"]
        if additional_context:
            prompt += f"\n\nAdditional context: {additional_context}"
        
        # Get COT template
        cot_steps = template["cot_template"].copy()
        
        return prompt, cot_steps
    
    def call_gpt4v_safety(self, image_path: str, prompt: str, expected_cot_steps: List[str], safety_category: str) -> Dict[str, Any]:
        """Call GPT-4V API to generate safety-focused COT reasoning response."""
        base64_image = self.encode_image(image_path)
        
        system_prompt = f"""You are an expert content safety analyst with deep understanding of harmful, inappropriate, and disgusting content detection.

Your task is to analyze multimodal content (images and text) for safety violations with detailed reasoning.

Safety Categories to Consider:
- Disgusting/revolting visual content
- Graphic violence or disturbing imagery  
- Inappropriate or harmful text-image combinations
- Content that violates platform safety policies
- Material that could cause psychological harm

For each analysis:
1. Provide exactly {len(expected_cot_steps)} clear reasoning steps
2. Be specific about what makes content safe or unsafe
3. Consider both obvious and subtle safety violations
4. Evaluate potential harm to different audiences
5. Conclude with a clear safety determination

Current focus: {safety_category}

Format your response as:
**Step 1:** [Initial content analysis]
**Step 2:** [Specific safety factor identification]
**Step 3:** [Risk and harm assessment]
**Step 4:** [Context and audience considerations]
**Step 5:** [Final safety determination]

**Safety Label:** [SAFE/UNSAFE]
**Confidence:** [1-10 scale]
**Reasoning:** [Brief summary of key safety factors]"""
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt}\n\nExpected reasoning structure:\n" + 
                               "\n".join([f"{i+1}. {step}" for i, step in enumerate(expected_cot_steps)])
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=messages,
                max_tokens=2048,
                temperature=0.7
            )
            
            return {
                "success": True,
                "content": response.choices[0].message.content,
                "usage": response.usage
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "content": None
            }
    
    def parse_safety_cot_response(self, response_content: str) -> Tuple[List[str], str, str, float]:
        """Parse COT steps, safety label, confidence, and reasoning from GPT-4V response."""
        lines = response_content.split('\n')
        cot_steps = []
        safety_label = "UNKNOWN"
        confidence_score = 5.0
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
                if "SAFE" in label_text.upper():
                    safety_label = "SAFE"
                elif "UNSAFE" in label_text.upper():
                    safety_label = "UNSAFE"
            
            # Look for confidence score
            elif line.startswith("**Confidence:**"):
                conf_text = line.split(":**", 1)[1].strip()
                try:
                    # Extract numeric confidence
                    import re
                    conf_match = re.search(r'(\d+(?:\.\d+)?)', conf_text)
                    if conf_match:
                        confidence_score = float(conf_match.group(1))
                        # Normalize to 0-1 if given as 1-10 scale
                        if confidence_score > 1:
                            confidence_score = confidence_score / 10.0
                except:
                    confidence_score = 0.5
            
            # Look for reasoning summary
            elif line.startswith("**Reasoning:**"):
                final_reasoning = line.split(":**", 1)[1].strip()
        
        # Fallback parsing if structured format not found
        if not cot_steps:
            step_markers = ["Step ", "step ", "1.", "2.", "3.", "4.", "5."]
            for line in lines:
                if any(marker in line for marker in step_markers) and len(line.split()) > 3:
                    cot_steps.append(line.strip())
        
        return cot_steps, safety_label, final_reasoning, confidence_score
    
    def quality_check_safety_sample(self, sample: Dict[str, Any]) -> bool:
        """Perform quality check on generated safety sample."""
        # Check if COT has reasonable number of steps
        if not (5 <= len(sample["cot_steps"]) <= 10):
            return False
        
        # Check if safety label is valid
        if sample["safety_label"] not in ["SAFE", "UNSAFE"]:
            return False
        
        # Check if confidence score is reasonable
        if not (0 <= sample["confidence_score"] <= 1):
            return False
        
        # Check if each step has reasonable length
        for step in sample["cot_steps"]:
            if len(step.split()) < 5:  # At least 5 words per step
                return False
        
        # Check if final reasoning exists and is substantial
        if not sample["final_decision"] or len(sample["final_decision"].split()) < 3:
            return False
        
        return True
    
    def generate_batch_1_safety_data(self, image_files: List[str], output_path: str) -> List[Dict[str, Any]]:
        """
        Focus: High-quality safety COT with manual quality review
        """
        batch_config = self.config['data_generation']['batch_1']
        target_samples = batch_config['target_samples']
        quality_check_samples = batch_config['quality_check_samples']
        safety_categories = batch_config['safety_categories']
        
        generated_samples = []
        quality_checked_count = 0
        
        print(f"🚀 Starting Batch 1 safety data generation: {target_samples} samples with GPT-4V")
        print(f"🛡️  Safety categories: {', '.join(safety_categories)}")
        print(f"📋 Quality check for first {quality_check_samples} samples")
        
        for i in tqdm(range(target_samples), desc="Generating safety COT samples"):
            # Select random image and safety template
            image_path = random.choice(image_files) if image_files else ""
            
            # Select safety category and corresponding template
            category = random.choice(safety_categories)
            template = next((t for t in self.safety_templates if t["category"] == category), 
                          self.safety_templates[0])
            
            # Generate safety prompt and expected COT
            prompt, expected_cot = self.generate_safety_prompt(template)
            
            # Call GPT-4V for safety analysis
            response = self.call_gpt4v_safety(image_path, prompt, expected_cot, category)
            
            if not response["success"]:
                print(f"❌ Failed to generate safety sample {i+1}: {response['error']}")
                continue
            
            # Parse safety response
            cot_steps, safety_label, final_reasoning, confidence = self.parse_safety_cot_response(response["content"])
            
            # Create safety sample
            sample = {
                "id": f"safety_batch1_{i+1:04d}",
                "image_path": image_path,
                "text_prompt": prompt,
                "safety_label": safety_label,
                "safety_category": category,
                "cot_steps": cot_steps,
                "confidence_score": confidence,
                "final_decision": final_reasoning,
                "template_used": template["category"],
                "generation_timestamp": time.time(),
                "model": "gpt-4v-safety",
                "batch": 1
            }
            
            # Quality check for first N samples
            if quality_checked_count < quality_check_samples:
                if self.quality_check_safety_sample(sample):
                    sample["quality_approved"] = True
                    quality_checked_count += 1
                else:
                    sample["quality_approved"] = False
                    print(f"⚠️  Quality check failed for safety sample {i+1}")
            
            generated_samples.append(sample)
            
            # Rate limiting
            time.sleep(1)  # 1 second between requests
            
            # Save intermediate results every 100 samples
            if (i + 1) % 100 == 0:
                intermediate_path = f"{output_path}_safety_batch1_intermediate_{i+1}.json"
                with open(intermediate_path, 'w') as f:
                    json.dump(generated_samples, f, indent=2)
                print(f"💾 Saved intermediate safety results: {len(generated_samples)} samples")
        
        # Save final results
        final_output = f"{output_path}_safety_batch1_final.json"
        with open(final_output, 'w') as f:
            json.dump(generated_samples, f, indent=2)
        
        # Generate summary statistics
        quality_approved = sum(1 for s in generated_samples if s.get("quality_approved", False))
        avg_cot_length = sum(len(s["cot_steps"]) for s in generated_samples) / len(generated_samples)
        safety_distribution = {}
        for sample in generated_samples:
            label = sample["safety_label"]
            safety_distribution[label] = safety_distribution.get(label, 0) + 1
        
        summary = {
            "batch": 1,
            "task": "safety_detection",
            "total_samples": len(generated_samples),
            "quality_approved": quality_approved,
            "quality_approval_rate": quality_approved / quality_check_samples if quality_check_samples > 0 else 0,
            "avg_cot_length": avg_cot_length,
            "safety_label_distribution": safety_distribution,
            "generation_date": time.strftime("%Y-%m-%d"),
            "model_used": "gpt-4v-safety"
        }
        
        with open(f"{output_path}_safety_batch1_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Safety Batch 1 generation complete!")
        print(f"📊 Total samples: {len(generated_samples)}")
        print(f"🎯 Quality approved: {quality_approved}/{quality_check_samples}")
        print(f"📏 Average COT length: {avg_cot_length:.1f} steps")
        print(f"🛡️  Safety distribution: {safety_distribution}")
        
        return generated_samples

def main():
    print("🛡️  LLaVA-OneVision Safety COT Data Generation")
    print("=" * 60)
    
    # Initialize generator
    generator = SafetyCOTDataGenerator()
    
    # Setup paths
    data_dir = Path("experiments/data_generation")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Image files - leave empty as requested, to be filled with actual paths
    image_files = []  # Will be populated with actual unsafe/safe content paths
    
    output_path = data_dir / "safety_cot_data"
    
    # Generate Batch 1 safety data (milestone)
    try:
        samples = generator.generate_batch_1_safety_data(image_files, str(output_path))
        print(f"\n🎉 Successfully generated {len(samples)} safety COT samples for Batch 1!")
        
    except Exception as e:
        print(f"❌ Error during safety generation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())