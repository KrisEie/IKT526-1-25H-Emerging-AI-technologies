import os
import json
import torch
import math
import random
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import config

def load_test_data():
    # Re-implement split logic to ensure consistency
    print("Loading data from:", config.DATA_PATH)
    with open(config.DATA_PATH, "r") as f:
        data = json.load(f)
    
    random.seed(config.SEED)
    random.shuffle(data)
    
    train_end = config.TRAIN_COUNT
    val_end = train_end + config.VAL_COUNT
    test_end = val_end + config.TEST_COUNT
    
    test_data = data[val_end:test_end]
    print(f"Loaded {len(test_data)} test samples.")
    return test_data

def calculate_perplexity(model, tokenizer, text, prompt):
    # Tokenize full text
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)
    
    # Tokenize prompt to find its length
    prompt_encodings = tokenizer(prompt, return_tensors="pt")
    prompt_len = prompt_encodings.input_ids.shape[1]
    
    # Create labels: clone input_ids and mask the prompt part
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100  # Mask prompt tokens
    
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        
    return torch.exp(loss).item()

def calculate_f1(prediction, ground_truth):
    pred_tokens = prediction.lower().split()
    truth_tokens = ground_truth.lower().split()
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0, 0.0, 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    return precision, recall, f1

def format_prompt(sample):
    instruction = sample["instruction"]
    input_text = sample.get("input", "")
    if input_text:
        return f"Instruction:\n{instruction}\n\nInput:\n{input_text}\n\nOutput:\n"
    else:
        return f"Instruction:\n{instruction}\n\nOutput:\n"

def main():
    # 1. Load Data
    test_data = load_test_data()
    
    # 2. Load Models (Base & Fine-tuned)
    print("Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        token=config.HF_TOKEN,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, token=config.HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading Fine-tuned Model...")
    # Load adapter
    ft_model = PeftModel.from_pretrained(base_model, os.path.join(config.OUTPUT_DIR, "best_model"))
    
    # 3. Evaluation Loop
    results = []
    total_ppl = 0
    total_f1 = 0
    
    # Select 10 random samples for detailed reporting as requested
    eval_samples = random.sample(test_data, 10)
    
    print("Starting Evaluation...")
    for i, sample in enumerate(eval_samples):
        prompt = format_prompt(sample)
        ground_truth = sample["output"]
        
        inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)
        
        # Generate
        with torch.no_grad():
            outputs = ft_model.generate(
                **inputs, 
                max_new_tokens=100, 
                do_sample=False # Greedy for eval
            )
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part (remove prompt)
        generated_text = prediction[len(prompt):].strip()
        
        # Metrics
        ppl = calculate_perplexity(ft_model, tokenizer, prediction, prompt)
        
        # F1
        prec, rec, f1 = calculate_f1(generated_text, ground_truth)
        
        total_ppl += ppl
        total_f1 += f1
        
        results.append({
            "instruction": sample["instruction"],
            "input": sample.get("input", ""),
            "ground_truth": ground_truth,
            "generated_output": generated_text,
            "perplexity": ppl,
            "f1_score": f1
        })
        
        print(f"Sample {i+1}: PPL={ppl:.2f}, F1={f1:.2f}")

    avg_ppl = total_ppl / len(eval_samples)
    avg_f1 = total_f1 / len(eval_samples)
    
    print(f"\nAverage Perplexity: {avg_ppl:.2f}")
    print(f"Average F1 Score: {avg_f1:.2f}")
    
    # Save Results
    output_file = os.path.join(config.GENERATIONS_DIR, "test_set_evaluation.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
