import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from torch.optim import SGD
import json
import argparse
from seq_eval import evaluate

# Define argparse to accept only hyperparameters: epochs, batch_size, and learning_rate
def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a model with specific hyperparameters.")
    parser.add_argument('--epochs', type=int, default=2, help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=30, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=5e-3, help="Learning rate for optimizer")
    parser.add_argument('--eval_on_train', action='store_true')

    return parser.parse_args()


# Load the arguments from command line
args = parse_args()

# Load dataset
with open("d2p_each_dataset.json", 'r') as f:
    dataset = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer.pad_token = tokenizer.eos_token

# Create result filename based on parameters
result_filename = f"./results/evaluation_results_epoch{args.epochs}_batchsize{args.batch_size}_lr{args.learning_rate}.jsonl"


if args.eval_on_train:
    result_filename = f"./results2/evaluation_results_epoch{args.epochs}_batchsize{args.batch_size}_lr{args.learning_rate}.jsonl"


# Open the .jsonl file in append mode to store results
with open(result_filename, 'a') as jsonl_file:

    test_datasets = {}
    train_datasets = {}
    for idx, individual_data in enumerate(dataset):
        print(f"Training on dataset {idx + 1}")
        
        # Prepare the train and test data
        train_data = individual_data['train']
        test_data = individual_data['test']
        test_datasets[idx] = test_data
        train_datasets[idx] = train_data
        # Convert to Hugging Face Dataset (directly use 'text' field)
        train_dataset = Dataset.from_dict({
            'text': [item['prompt'] + item['completion'] for item in train_data]
        })
        print(train_dataset)
        print(train_dataset[0])
        
        # Training configuration
        training_args = SFTConfig(
            output_dir="/tmp",   
            num_train_epochs=args.epochs,                      # Use dynamic epoch count
            per_device_train_batch_size=args.batch_size,        # Use dynamic batch size
            max_seq_length=128,  # Max sequence length (hardcoded)
            logging_steps=100,
        )

        # Initialize SGD optimizer with dynamic learning rate
        optimizer = SGD(model.parameters(), lr=args.learning_rate)

        # Initialize SFTTrainer with the model, dataset, config, optimizer, and scheduler
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            args=training_args,  # Pass the training config
            optimizers=(optimizer, None)  # Pass only the optimizer (no scheduler for now)
        )

        # Train the model using the SFTTrainer
        trainer.train()

        #### Evaluate performance on all test datasets after training on this one
        for test_data_idx, test_data in test_datasets.items():
            generated_completions, accuracy = evaluate(model, tokenizer, test_data)

            # Prepare the result to be written as a JSON object
            result = {
                'train_data_idx': idx,
                'test_data_idx': test_data_idx,
                'accuracy': accuracy,
                'category': 'test'
            }
            
            # Write the result as a new line in the .jsonl file
            jsonl_file.write(json.dumps(result) + '\n')

            # Optionally, print the results
            print(f"Evaluating on Test Dataset {test_data_idx + 1} after training on Dataset {idx + 1}")
            print(f"Exact Match Accuracy: {accuracy * 100:.2f}%")

        if args.eval_on_train:
            #### Evaluate performance on all test datasets after training on this one
            for test_data_idx, test_data in train_datasets.items():
                generated_completions, accuracy = evaluate(model, tokenizer, test_data)

                # Prepare the result to be written as a JSON object
                result = {
                    'train_data_idx': idx,
                    'test_data_idx': test_data_idx,
                    'accuracy': accuracy,
                    'category': 'train'
                }
                
                # Write the result as a new line in the .jsonl file
                jsonl_file.write(json.dumps(result) + '\n')

                # Optionally, print the results
                print(f"Evaluating on Train Dataset {test_data_idx + 1} after training on Dataset {idx + 1}")
                print(f"Exact Match Accuracy: {accuracy * 100:.2f}%")


        # Clear GPU memory after each dataset
        torch.cuda.empty_cache()
