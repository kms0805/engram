import argparse
import os
from datetime import datetime
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from torch.optim import SGD
import json
import matplotlib.pyplot as plt

# Argument Parsing
parser = argparse.ArgumentParser(description="Training and Evaluation Configuration")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
args = parser.parse_args()

# Parameters from argparse
epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate

# Generate output folder and result folder name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = "./output"
os.makedirs(output_folder, exist_ok=True)
results_name = f"{timestamp}_epochs{epochs}_bs{batch_size}_lr{learning_rate}"
result_path = os.path.join(output_folder, results_name)
os.makedirs(result_path, exist_ok=True)
models_path = os.path.join(result_path, "models")
os.makedirs(models_path, exist_ok=True)

# File paths
dataset_path = "d2p_each_dataset.json"

# Load Dataset
with open(dataset_path, 'r') as f:
    dataset = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Model and Tokenizer
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Freeze the embedding layer (embed_tokens)
for param in model.model.embed_tokens.parameters():
    param.requires_grad = False

# Freeze the head (lm_head)
for param in model.lm_head.parameters():
    param.requires_grad = False

# Set the padding token
tokenizer.pad_token = tokenizer.eos_token

# Tokenization Function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

# Prepare Train and Test Datasets
train_datasets = []
test_datasets = []

for idx, individual_data in enumerate(dataset):    
    # Prepare Train and Test Data
    train_data = individual_data['train']
    test_data = individual_data['test']

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_dict({
        'text': [item['prompt'] + item['completion'] for item in train_data]
    }).map(tokenize_function, batched=True)
    
    test_dataset = Dataset.from_dict({
        'text': [item['prompt'] + item['completion'] for item in test_data]
    }).map(tokenize_function, batched=True)

    train_datasets.append(train_dataset)
    test_datasets.append(test_dataset)

# Store evaluation results for plotting
series_data = []

# Training and Evaluation Loop
for train_idx, train_dataset in enumerate(train_datasets):
    # Training Configuration
    training_args = SFTConfig(
        output_dir=f"./outputs/train_{train_idx}",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        max_seq_length=128,
        logging_steps=100,
    )

    # Initialize Optimizer
    optimizer = SGD(model.parameters(), lr=learning_rate)

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        args=training_args,
        optimizers=(optimizer, None)
    )

    # Train the Model
    trainer.train()
    
    # Save the trained model for current train_idx
    model_save_path = os.path.join(models_path, f"model_train_{train_idx}")
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Evaluate the Model
    for test_idx, test_dataset in enumerate(test_datasets):
        eval_results = trainer.evaluate(eval_dataset=test_dataset)
        eval_loss = eval_results['eval_loss']

        # Append data for graph
        series_data.append({
            "train_idx": train_idx,
            "test_idx": test_idx,
            "eval_loss": eval_loss,
        })

    # Clear GPU Memory
    torch.cuda.empty_cache()

# Plotting after all evaluations are done
plt.figure(figsize=(12, 8))

# Plot each test_idx's line
test_idx_set = set(item["test_idx"] for item in series_data)

for test_idx in test_idx_set:
    test_data = [item for item in series_data if item["test_idx"] == test_idx]
    plt.plot(
        [item["train_idx"] for item in test_data],
        [item["eval_loss"] for item in test_data],
        marker='o',
        label=f'Test {test_idx}'
    )

# Set plot labels and title
plt.xlabel("Train Index")
plt.ylabel("Evaluation Loss")
plt.title("Evaluation Loss vs Train Index for All Tests")
plt.legend()
plt.grid(True)

# Save the plot
plot_path = os.path.join(result_path, "loss_plot.png")
plt.savefig(plot_path)
plt.show()

# Save series data
series_data_path = os.path.join(result_path, "series_data.json")
with open(series_data_path, 'w') as f:
    json.dump(series_data, f, indent=4)

print(f"Results saved to {result_path}")
