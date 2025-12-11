import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import config

def get_novel_instructions():
    return [
        "Write a haiku about artificial intelligence.",
        "Explain quantum computing to a 5-year-old.",
        "Write a Python function to check if a number is prime.",
        "What are the three main laws of robotics according to Asimov?",
        "Summarize the benefits of renewable energy in one sentence.",
        "Translate 'Hello, how are you?' into French, Spanish, and German.",
        "Brainstorm 3 unique names for a coffee shop.",
        "Act as a travel guide and recommend 3 places to visit in Tokyo.",
        "Solve this riddle: I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I?",
        "Why is the sky blue?"
    ]

def generate_response(model, tokenizer, instruction, strategy):
    prompt = f"Instruction:\n{instruction}\n\nOutput:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    generation_kwargs = {
        "max_new_tokens": 150,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    if strategy == "Greedy Decoding":
        generation_kwargs["do_sample"] = False
    elif strategy == "Temperature Sampling":
        generation_kwargs["temperature"] = 0.7
        generation_kwargs["top_k"] = 0 # Disable top_k to rely on temp
    elif strategy == "Nucleus Sampling":
        generation_kwargs["top_p"] = 0.9
        generation_kwargs["temperature"] = 1.0 # Standard for nucleus
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
        
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction[len(prompt):].strip()

def main():
    # 1. Load Model
    print("Loading Fine-tuned Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        token=config.HF_TOKEN,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, token=config.HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = PeftModel.from_pretrained(base_model, os.path.join(config.OUTPUT_DIR, "best_model"))
    
    # 2. Define Strategies
    strategies = [
        "Greedy Decoding",
        "Temperature Sampling",
        "Nucleus Sampling"
    ]
    
    instructions = get_novel_instructions()
    results = []
    
    print("Starting Generation...")
    for i, instruction in enumerate(instructions):
        print(f"Processing Instruction {i+1}/{len(instructions)}...")
        
        for strategy in strategies:
            output = generate_response(model, tokenizer, instruction, strategy)
            
            results.append({
                "input_instruction": instruction,
                "model_output": output,
                "strategy_used": strategy,
                "scores": {} # Placeholder for manual scoring
            })
            
    # 3. Save Results
    output_file = os.path.join(config.GENERATIONS_DIR, "sampling_comparison.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
