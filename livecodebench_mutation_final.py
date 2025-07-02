import os
import json
import time
from mutations.Mutation import BaseMutation
import argparse

def load_livecodebench_dataset(jsonl_file="livecodebench_mutation.jsonl"):
    """Load LiveCodeBench dataset from JSONL file and format for mutations (EXACTLY like BigCodeBench)."""
    if not os.path.exists(jsonl_file):
        raise FileNotFoundError(f"LiveCodeBench dataset not found at {jsonl_file}")
    
    dataset = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                # Already in the right format (task_id, question, starter_code)
                dataset.append(item)
    
    return dataset

def generate_mutations(model_name='gpt-4o', start_idx=None, end_idx=None, dataset_name='livecodebench'):
    """Generate mutations for LiveCodeBench dataset and save to mutation_output directory (EXACTLY like BigCodeBench)."""
    # Create output directory if it doesn't exist
    os.makedirs('mutation_output', exist_ok=True)
    
    # Load LiveCodeBench dataset
    dataset = load_livecodebench_dataset()
    
    # Slice dataset if specified
    if start_idx is not None and end_idx is not None:
        dataset = dataset[start_idx:end_idx]
        print(f"Processing subset of dataset: {start_idx} to {end_idx} (total: {len(dataset)} items)")
    else:
        print(f"Processing full dataset ({len(dataset)} items)")
    
    # Define all mutation types with their prompts and output paths (EXACTLY SAME AS BIGCODEBENCH)
    initialization_message = "You are given the following sentence. Your task is to modify it according to the instructions provided. Keep the overall meaning, mathematical notation, and formatting intact. DO NOT modify function signatures, parameter types, or return types. You do not need to provide any code or implementation details."
    finalization_message = "ONlY return final prompt, do not return any other text or any implementation"
    mutation_types = [
        {
            "name": "original",
            "prompt": "Return the same sentence as the input: {question}",
            "output_path": f"mutation_output/original_{model_name}_{dataset_name}_final.jsonl"
        },
        {
            "name": "active_to_passive",
            "prompt": initialization_message +  "Change the following sentence to passive voice: {question}" + finalization_message,
            "output_path": f"mutation_output/active_to_passive_{model_name}_{dataset_name}_final.jsonl"
        },
        {
            "name": "declarative_to_interrogative",
            "prompt": initialization_message +  "Change the following sentence to interrogative voice: {question}" + finalization_message,
            "output_path": f"mutation_output/declarative_to_interrogative_{model_name}_{dataset_name}_final.jsonl"
        },
        {
            "name": "verb_to_similar_verb",
            "prompt": initialization_message +  "Change the verbs of the following sentence to a similar verb: {question}" + finalization_message,
            "output_path": f"mutation_output/verb_to_similar_verb_{model_name}_{dataset_name}.jsonl"
        },
        {
            "name": "lowercase_to_uppercase",
            "prompt": initialization_message +  "Change the following sentence to uppercase: {question}" + finalization_message,
            "output_path": f"mutation_output/lowercase_to_uppercase_{model_name}_{dataset_name}_final.jsonl"
        },
        {
            "name": "rephrase_prompt",
            "prompt":  initialization_message + "Rephrase the following sentence: {question}" + finalization_message,
            "output_path": f"mutation_output/rephrase_prompt_{model_name}_{dataset_name}_final.jsonl"
        },
        {
            "name": "task_function_name",
            "prompt": initialization_message + "Modify the following sentence to refer to a function with a descriptive name that represents the task: {question}" + finalization_message,
            "output_path": f"mutation_output/task_function_name_{model_name}_{dataset_name}_final.jsonl"
        },
        {
            "name": "adversarial_function_name",
            "prompt": initialization_message + "Modify the following sentence to refer to a function with a misleading name that doesn't represent the actual task: {question}" + finalization_message,
            "output_path": f"mutation_output/adversarial_function_name_{model_name}_{dataset_name}_final.jsonl"
        }
    ]
    
    # Run each mutation
    for mutation_type in mutation_types:
        print(f"\nGenerating mutations for: {mutation_type['name']}")
        
        # Create mutation instance
        mutation = BaseMutation(
            prompt=mutation_type["prompt"] + "ONlY return the Modified prompt, do not return any other text or any implementation",
            dataset=dataset,
            output_path=mutation_type["output_path"],
            model=model_name
        )
        
        # Run the mutation
        start_time = time.time()
        mutation.run()
        elapsed_time = time.time() - start_time
        
        print(f"Completed {mutation_type['name']} in {elapsed_time:.2f} seconds")
        print(f"Results saved to {mutation_type['output_path']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mutations for LiveCodeBench dataset")
    parser.add_argument("--model", default="gpt-4o", help="Model to use for mutations")
    parser.add_argument("--start", type=int, help="Start index in dataset")
    parser.add_argument("--end", type=int, help="End index in dataset")
    parser.add_argument("--dataset", default="livecodebench", help="Dataset name for output files")
    args = parser.parse_args()
    
    generate_mutations(model_name=args.model, start_idx=args.start, end_idx=args.end, dataset_name=args.dataset)