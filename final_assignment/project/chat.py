import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import config

def main():
    print("Loading model... (this may take a minute)")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        token=config.HF_TOKEN,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, token=config.HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = PeftModel.from_pretrained(base_model, os.path.join(config.OUTPUT_DIR, "best_model"))
    
    print("\n" + "="*50)
    print("🤖 Llama 3.2 1B LoRA Chatbot")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        prompt = f"Instruction:\n{user_input}\n\nOutput:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = prediction[len(prompt):].strip()
        
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    main()
