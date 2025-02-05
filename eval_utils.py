import torch

def exact_match_accuracy(generated_completions, target_completions, prompts):
    correct_count = 0 
    for gen, target, prompt in zip(generated_completions, target_completions, prompts):
        # Adjust the target to only the first two words if it contains more than two words(first name, last name)
        target_words = target.split()
        if len(target_words) > 2:
            target = " ".join(target_words[:2])

        # Remove the prompt from the generated completion
        generated_part = gen[len(prompt):]  # Take everything after the prompt

        # Check if the generated part starts with the target completion (case-insensitive)
        if generated_part.strip().lower().startswith(target.strip().lower()):
            correct_count += 1
    
    # Calculate accuracy as the percentage of correct matches
    accuracy = correct_count / len(generated_completions) if len(generated_completions) > 0 else 0.0
    return accuracy



# Function to evaluate the model on the test data
def evaluate_QA(model, tokenizer, test_dataset):
    model.eval()
    test_prompts = [item['prompt'] for item in test_dataset]
    test_completions = [item['completion'] for item in test_dataset]

    generated_completions = []
    
    for prompt in test_prompts:
        # Tokenize each prompt separately without padding
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

        # Generate completion one by one (no padding)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=64,
            )

        # Decode the generated completion
        generated_completions.append(tokenizer.decode(output[0], skip_special_tokens=True))

    # Evaluate the accuracy based on whether the generated part starts with the target completion
    accuracy = exact_match_accuracy(generated_completions, test_completions, test_prompts)

    return generated_completions, accuracy