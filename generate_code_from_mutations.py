import json
import copy
import tqdm
import concurrent.futures as cfuts
import random
import os
import argparse
import model
from lcb_runner.utils.extraction_utils import extract_code
from lcb_runner.lm_styles import LMStyle


class MockCodeGenerationProblem:
    """Mock class to simulate CodeGenerationProblem for prompting"""
    def __init__(self, question_content, starter_code=""):
        self.question_content = question_content
        self.starter_code = starter_code


def get_lcb_prompt_template(question_content, starter_code="", is_test_case=False, public_test_case=None):
    """Create LiveCodeBench-style prompt using their template format"""
    # Create mock problem object
    problem = MockCodeGenerationProblem(question_content, starter_code)
    
    if is_test_case:
        # Test case generation prompt
        prompt = f"### Question:\n{problem.question_content}\n\n"
        
        if public_test_case:
            prompt += "### Example Test Case:\n"
            prompt += f"Input:\n{public_test_case['input']}\n"
            prompt += f"Expected Output:\n{public_test_case['output']}\n\n"
        
        prompt += "### Task:\n"
        prompt += "Generate 5 test cases for this problem. Return test cases as a Python list where each test case is a dictionary with 'input' and 'expected_output' keys.\n\n"
        prompt += "### Format:\n"
        prompt += "```python\n"
        prompt += "["
        prompt += "    {"
        prompt += "        'input': 'your_input_here',\n"
        prompt += "        'expected_output': 'expected_output_here'\n"
        prompt += "    },"
        prompt += "    # Add more test cases..."
        prompt += "]"
        prompt += "```\n\n"
        prompt += "### Requirements:\n"
        prompt += "- Generate at least 5 comprehensive test cases\n"
        prompt += "- Include edge cases (empty inputs, boundary values, etc.)\n"
        prompt += "- Cover different scenarios mentioned in the problem\n"
        prompt += "- Ensure input/output format matches the problem specification\n"
        prompt += "- Return the test_cases list as shown in the format above\n\n"
        prompt += "### Answer: (use the provided format with backticks)\n\n"
        
    else:
        # Code generation prompt (original)
        prompt = f"### Question:\n{problem.question_content}\n\n"
        
        if problem.starter_code:
            prompt += "### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
            prompt += f"```python\n{problem.starter_code}\n```\n\n"
        else:
            prompt += "### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
            prompt += "```python\n# YOUR CODE HERE\n```\n\n"
        
        prompt += "### Answer: (use the provided format with backticks)\n\n"
    
    return prompt


def form_messages(problems, is_test_case=False):
    """Form messages using LiveCodeBench format"""
    if is_test_case:
        system_message = (
            "You are an expert test case generator for programming problems. "
            "Generate 5 test cases that thoroughly validate solutions. "
            "Return test cases as a Python list with 'input' and 'expected_output' keys. "
            "Include edge cases, boundary conditions, and diverse scenarios. "
            "Ensure all test cases follow the exact input/output format specified in the problem."
        )
    else:
        system_message = (
            "You are an expert Python programmer. You will be given a question (problem specification) "
            "and will generate a correct Python program that matches the specification and passes all tests."
        )
    
    return [
        (
            [
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': problem[0]}
            ],
            problem[1]  # task_id
        )
        for problem in problems
    ]

def run_model(message, model_name='o4-mini'):
    if model_name.startswith('deepseek'):
        return model.call_deepseek(message)
    elif model_name.startswith("gemini"):
        return model.call_gemini(message)
    else:
        return model.call_chat_gpt(message, model=model_name)

def run(messages, output_path, model_name='o4-mini'):
    def process_message(message, task_id):
        response, prompt_tokens, completion_tokens = run_model(message, model_name)
        print(f"Processed {task_id}")
        # Use LCB's code extraction with OpenAIChat style (works for most models)
        code = extract_code(response, LMStyle.OpenAIChat)
        return {
            'task_id': task_id,
            'prompt': message,
            'response': response,
            'response_code': code
        }

    with cfuts.ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(process_message, message[0], message[1]) 
            for message in messages
        ]
        
        responses = []
        for future in tqdm.tqdm(cfuts.as_completed(futures), total=len(futures)):
            responses.append(future.result())

    # Sort by task_id for consistent output
    responses.sort(key=lambda x: x['task_id'])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for res in responses:
            f.write(json.dumps(res) + '\n')
    
    print(f"Generated {len(responses)} responses saved to {output_path}")


def load_test_case_data(test_case_file="extracted_test_cases.jsonl"):
    """Load test case data for problems"""
    test_case_data = {}
    
    if os.path.exists(test_case_file):
        print(f"Loading test case data from {test_case_file}")
        with open(test_case_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                test_case_data[data['id']] = data
        print(f"Loaded test case data for {len(test_case_data)} problems")
    else:
        print(f"Test case file {test_case_file} not found. Proceeding without test case examples.")
    
    return test_case_data


def run_all_mutations(mutation_dir, output_dir, model_name, mutation_model, dataset, test_case, max_workers, test_case_file=None):
    """Run all available mutations with the specified model"""
    if not os.path.exists(mutation_dir):
        print(f"Error: Mutation directory not found: {mutation_dir}")
        return
    
    # Load test case data if generating test cases
    test_case_data = {}
    if test_case and test_case_file:
        test_case_data = load_test_case_data(test_case_file)
    
    # Get all mutation files
    mutation_files = [f for f in os.listdir(mutation_dir) if f.endswith('.jsonl')]
    
    if not mutation_files:
        print(f"No mutation files found in {mutation_dir}")
        return
    
    print(f"Found {len(mutation_files)} mutation files:")
    for f in mutation_files:
        print(f"  {f}")
    
    type_suffix = 'test' if test_case else 'code'
    
    for mutation_file in mutation_files:
        # Extract mutation type from filename
        mutation_type = mutation_file.replace(f'_{mutation_model}_{dataset}_final.jsonl', '')
        
        print(f"\n{'='*60}")
        print(f"Processing: {mutation_type}")
        print(f"{'='*60}")
        
        mutation_path = os.path.join(mutation_dir, mutation_file)
        output_path = f"{output_dir}/{type_suffix}_{mutation_type}_{model_name}_{dataset}_final.jsonl"
        
        # Skip if output already exists
        if os.path.exists(output_path):
            print(f"Output already exists: {output_path}")
            print("Skipping...")
            continue
        
        # Load mutation data
        print(f"Loading mutations from {mutation_path}")
        with open(mutation_path) as f:
            mutations = [json.loads(line) for line in f]
        
        print(f"Loaded {len(mutations)} mutations")
        
        # Create problems from mutations using LCB format
        problems = []
        for mutation in mutations:
            # Check if this is a mutated question or original
            if 'mutation' in mutation:
                # This is from mutation output
                question = mutation['mutation']
            elif 'question' in mutation:
                # This might be original data
                question = mutation['question']
            else:
                print(f"Warning: No question found in mutation {mutation.get('task_id', 'unknown')}")
                continue
                
            task_id = mutation['task_id']
            starter_code = mutation.get('starter_code', '')
            
            # Get test case example if available
            public_test_case = None
            if args.test_case and task_id in test_case_data:
                test_data = test_case_data[task_id]
                if test_data.get('public_test_cases') and len(test_data['public_test_cases']) > 0:
                    first_test = test_data['public_test_cases'][0]
                    public_test_case = {
                        'input': first_test['input'].strip(),
                        'output': first_test['output'].strip()
                    }
            
            # Create LiveCodeBench-style template
            formatted_prompt = get_lcb_prompt_template(question, starter_code, args.test_case, public_test_case)
            problems.append((formatted_prompt, task_id))
        
        print(f"Created {len(problems)} problems for generation")
        
        if len(problems) == 0:
            print("No valid problems found. Skipping.")
            continue
        
        # Generate messages
        messages = form_messages(problems, test_case)
        
        # Run generation
        print(f"Generating {'test cases' if test_case else 'code'} with {max_workers} workers...")
        temp_response = "```python\n print('ali')```"
        def run_with_workers(messages, output_path, max_workers, model_name):
            def process_message(message, task_id):
                try:
                    response, prompt_tokens, completion_tokens = run_model(message, model_name)
                except Exception as e:
                    print(e)
                    response, prompt_tokens, completion_tokens = temp_response, 0, 0
                print(f"Processed {task_id}")
                # Use LCB's code extraction with OpenAIChat style (works for most models)
                code = extract_code(response, LMStyle.OpenAIChat)
                return {
                    'task_id': task_id,
                    'prompt': message,
                    'response': response,
                    'response_code': code
                }

            with cfuts.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_message, message[0], message[1]) 
                    for message in messages
                ]
                
                responses = []
                for future in tqdm.tqdm(cfuts.as_completed(futures), total=len(futures)):
                    responses.append(future.result())

            # Sort by task_id for consistent output
            responses.sort(key=lambda x: x['task_id'])

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                for res in responses:
                    f.write(json.dumps(res) + '\n')
            
            print(f"Generated {len(responses)} responses saved to {output_path}")
        
        run_with_workers(messages, output_path, max_workers, model_name)
    
    print(f"\n{'='*60}")
    print("All mutations processed!")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate code from LiveCodeBench mutations")
    parser.add_argument("--mutation-type", default=None, help="Specific mutation type to use (if not specified, runs all)")
    parser.add_argument("--model", default="o4-mini", help="Model for code generation (use 'deepseek' for DeepSeek)")
    parser.add_argument("--mutation-model", default="gpt-4o", help="Model used for mutations")
    parser.add_argument("--dataset", default="livecodebench", help="Dataset name")
    parser.add_argument("--test-case", action="store_true", help="Generate test cases instead of code")
    parser.add_argument("--max-workers", type=int, default=32, help="Number of parallel workers")
    parser.add_argument("--mutation-dir", default="mutation_output", help="Directory containing mutation files")
    parser.add_argument("--output-dir", default="generation_output", help="Output directory")
    parser.add_argument("--run-all", action="store_true", help="Run all available mutations")
    parser.add_argument("--test-case-file", default="livecode_test.jsonl", help="File containing test case data for examples")
    
    args = parser.parse_args()
    
    random.seed(42)
    
    # If run-all is specified or no mutation-type is given, run all mutations
    if args.run_all or args.mutation_type is None:
        print("Running all available mutations...")
        run_all_mutations(
            args.mutation_dir, 
            args.output_dir, 
            args.model, 
            args.mutation_model, 
            args.dataset, 
            args.test_case, 
            args.max_workers,
            args.test_case_file if args.test_case else None
        )
    else:
        # Original single mutation logic
        type_suffix = 'test' if args.test_case else 'code'
        
        # Load test case data if generating test cases
        test_case_data = {}
        if args.test_case and args.test_case_file:
            test_case_data = load_test_case_data(args.test_case_file)
        
        # File paths
        mutation_file = f"{args.mutation_dir}/{args.mutation_type}_{args.mutation_model}_{args.dataset}_final.jsonl"
        output_path = f"{args.output_dir}/{type_suffix}_{args.mutation_type}_{args.model}_{args.dataset}_final.jsonl"
        
        # Check if mutation file exists
        if not os.path.exists(mutation_file):
            print(f"Error: Mutation file not found: {mutation_file}")
            print("Available mutation files:")
            if os.path.exists(args.mutation_dir):
                for f in os.listdir(args.mutation_dir):
                    if f.endswith('.jsonl'):
                        print(f"  {f}")
            exit(1)
        
        # Load mutation data
        print(f"Loading mutations from {mutation_file}")
        with open(mutation_file) as f:
            mutations = [json.loads(line) for line in f]
        
        print(f"Loaded {len(mutations)} mutations")
        
        # Create problems from mutations using LCB format
        problems = []
        for mutation in mutations:
            # Check if this is a mutated question or original
            if 'mutation' in mutation:
                # This is from mutation output
                question = mutation['mutation']
            elif 'question' in mutation:
                # This might be original data
                question = mutation['question']
            else:
                print(f"Warning: No question found in mutation {mutation.get('task_id', 'unknown')}")
                continue
                
            task_id = mutation['task_id']
            starter_code = mutation.get('starter_code', '')
            
            # Get test case example if available
            public_test_case = None
            if args.test_case and task_id in test_case_data:
                test_data = test_case_data[task_id]
                if test_data.get('public_test_cases') and len(test_data['public_test_cases']) > 0:
                    first_test = test_data['public_test_cases'][0]
                    public_test_case = {
                        'input': first_test['input'].strip(),
                        'output': first_test['output'].strip()
                    }
            
            # Create LiveCodeBench-style template
            formatted_prompt = get_lcb_prompt_template(question, starter_code, args.test_case, public_test_case)
            problems.append((formatted_prompt, task_id))
        
        print(f"Created {len(problems)} problems for generation")
        
        if len(problems) == 0:
            print("No valid problems found. Exiting.")
            exit(1)
        
        # Show first problem as example
        print("\nExample problem:")
        print("=" * 50)
        print(problems[0][0][:500] + "..." if len(problems[0][0]) > 500 else problems[0][0])
        print("=" * 50)
        
        # Generate messages
        messages = form_messages(problems, args.test_case)
        
        # Run generation
        print(f"\nGenerating {'test cases' if args.test_case else 'code'} with {args.max_workers} workers...")
        
        # Update the run function to use the specified number of workers and model
        def run_with_workers(messages, output_path, max_workers, model_name):
            def process_message(message, task_id):
                response, prompt_tokens, completion_tokens = run_model(message, model_name)
                print(f"Processed {task_id}")
                # Use LCB's code extraction with OpenAIChat style (works for most models)
                code = extract_code(response, LMStyle.OpenAIChat)
                return {
                    'task_id': task_id,
                    'prompt': message,
                    'response': response,
                    'response_code': code
                }

            with cfuts.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_message, message[0], message[1]) 
                    for message in messages
                ]
                
                responses = []
                for future in tqdm.tqdm(cfuts.as_completed(futures), total=len(futures)):
                    responses.append(future.result())

            # Sort by task_id for consistent output
            responses.sort(key=lambda x: x['task_id'])

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                for res in responses:
                    f.write(json.dumps(res) + '\n')
            
            print(f"Generated {len(responses)} responses saved to {output_path}")
        
        run_with_workers(messages, output_path, args.max_workers, args.model)