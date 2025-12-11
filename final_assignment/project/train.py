import os
import json
import time
import random
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import config

def print_gpu_utilization():
    if torch.cuda.is_available():
        print(f"GPU memory occupied: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Max GPU memory occupied: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    else:
        print("CUDA not available.")

def load_and_split_data():
    print("Loading data from:", config.DATA_PATH)
    with open(config.DATA_PATH, "r") as f:
        data = json.load(f)
    
    print(f"Total samples in dataset: {len(data)}")
    
    # Reproducible shuffle
    random.seed(config.SEED)
    random.shuffle(data)
    
    # Select subset
    train_end = config.TRAIN_COUNT
    val_end = train_end + config.VAL_COUNT
    test_end = val_end + config.TEST_COUNT
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:test_end]
    
    print(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Convert to Hugging Face Dataset
    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data)
    })
    
    return dataset

def format_instruction(sample):
    instruction = sample["instruction"]
    input_text = sample.get("input", "")
    output_text = sample["output"]
    
    if input_text:
        text = f"Instruction:\n{instruction}\n\nInput:\n{input_text}\n\nOutput:\n{output_text}"
    else:
        text = f"Instruction:\n{instruction}\n\nOutput:\n{output_text}"
        
    return {"text": text}

def main():
    start_time = time.time()
    
    # 1. Data Setup
    dataset = load_and_split_data()
    dataset = dataset.map(format_instruction)
    
    # 2. Model & Tokenizer
    print(f"Loading model: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, token=config.HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in 4-bit or 8-bit if needed, or fp16
    # Using fp16 for now as requested, assuming GPU can handle 1B model
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        token=config.HF_TOKEN,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 3. LoRA Configuration
    print(f"Configuring LoRA with rank {config.LORA_R}")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.TARGET_MODULES
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 4. Tokenization
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH
        )
        # Create labels and mask padding tokens
        model_inputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in input_id]
            for input_id in model_inputs["input_ids"]
        ]
        return model_inputs
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        num_train_epochs=config.NUM_EPOCHS,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
        report_to="none", # Disable wandb/mlflow
        seed=config.SEED,
        data_seed=config.SEED,
    )
    
    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
    )
    
    # 7. Train
    print("Starting training...")
    trainer.train()
    
    # 8. Save Adapter
    print("Saving best model adapter...")
    trainer.save_model(os.path.join(config.OUTPUT_DIR, "best_model"))
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*30)
    print("TRAINING COMPLETE")
    print(f"Total Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print_gpu_utilization()
    print("="*30 + "\n")

if __name__ == "__main__":
    main()
