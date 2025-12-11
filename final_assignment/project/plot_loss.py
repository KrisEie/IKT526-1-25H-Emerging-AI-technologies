import os
import json
import matplotlib.pyplot as plt
import config

def plot_loss():
    # Find the latest checkpoint folder to get the full history
    checkpoints = [d for d in os.listdir(config.OUTPUT_DIR) if d.startswith("checkpoint-")]
    if not checkpoints:
        print("No checkpoint folders found to read logs from.")
        return
    
    # Sort by step number
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
    log_path = os.path.join(config.OUTPUT_DIR, latest_checkpoint, "trainer_state.json")
    
    if not os.path.exists(log_path):
        print(f"Could not find trainer_state.json at {log_path}")
        return
        
    print(f"Reading logs from: {log_path}")
    with open(log_path, "r") as f:
        data = json.load(f)
        
    history = data["log_history"]
    
    train_steps = []
    train_loss = []
    val_steps = []
    val_loss = []
    
    for entry in history:
        if "loss" in entry:
            train_steps.append(entry["step"])
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            val_steps.append(entry["step"])
            val_loss.append(entry["eval_loss"])
            
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_loss, label="Training Loss", alpha=0.6)
    plt.plot(val_steps, val_loss, label="Validation Loss", marker='o', linewidth=2)
    
    plt.title("Training and Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    output_path = os.path.join(config.PLOTS_DIR, "loss_curves.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    plot_loss()
