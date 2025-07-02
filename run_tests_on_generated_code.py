#!/usr/bin/env python3
"""
Script to run test cases on generated code and check pass/fail status.
This script reads:
1. livecode_test.jsonl - contains problem IDs and public test cases
2. generation_output/ files - contains generated code for each problem
3. Executes tests and determines pass/fail status
"""

import json
import os
import sys
import signal
import time
import subprocess 
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from io import StringIO
import argparse


@dataclass
class TestResult:
    task_id: str
    passed: bool
    error_message: Optional[str] = None
    execution_time: Optional[float] = None


@dataclass
class TestCase:
    input_data: str
    expected_output: str


@dataclass
class Problem:
    task_id: str
    test_cases: List[TestCase]
    function_name: Optional[str] = None


def load_test_problems(jsonl_file: str) -> Dict[str, Problem]:
    """Load problems and test cases from livecode_test.jsonl"""
    problems = {}
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                task_id = data['id']
                
                # Extract test cases
                test_cases = []
                if 'public_test_cases' in data and data['public_test_cases']:
                    for test in data['public_test_cases']:
                        if 'input' in test and 'output' in test:
                            test_cases.append(TestCase(
                                input_data=test['input'].strip(),
                                expected_output=test['output'].strip()
                            ))
                
                problems[task_id] = Problem(
                    task_id=task_id,
                    test_cases=test_cases,
                    function_name=data.get('function_name')
                )
    
    return problems


def load_generated_code(generation_file: str) -> Dict[str, str]:
    """Load generated code from a JSONL file"""
    generated_code = {}
    
    if not os.path.exists(generation_file):
        print(f"Warning: Generation file {generation_file} not found")
        return generated_code
    
    with open(generation_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                task_id = data.get('task_id')
                
                # Try different possible fields for the code
                code = None
                if 'response_code' in data and data['response_code']:
                    code = data['response_code']
                elif 'response' in data:
                    # Extract code from response if it contains code blocks
                    response = data['response']
                    if '```python' in response:
                        # Extract code between ```python and ```
                        start = response.find('```python')
                        if start != -1:
                            start = response.find('\n', start) + 1
                            end = response.find('```', start)
                            if end != -1:
                                code = response[start:end].strip()
                    elif '```' in response:
                        # Extract code between ``` blocks
                        parts = response.split('```')
                        if len(parts) >= 3:
                            code = parts[1].strip()
                            # Remove language identifier if present
                            if code.startswith('python\n'):
                                code = code[7:]
                
                if code and task_id:
                    generated_code[task_id] = code
    
    return generated_code


def fix_function_name(code: str, expected_name: str) -> str:
    """Fix function name in generated code if it doesn't match expected name"""
    if not expected_name or 'def ' not in code:
        return code
    
    lines = code.split('\n')
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('def ') and '(' in stripped:
            # Extract current function name
            func_part = stripped[4:stripped.find('(')]
            if func_part != expected_name:
                # Replace function name
                lines[i] = line.replace(f'def {func_part}(', f'def {expected_name}(')
                break
    
    return '\n'.join(lines)


def run_single_test(code: str, test_case: TestCase, function_name: Optional[str] = None, timeout: int = 10) -> Tuple[bool, Optional[str]]:
    """Run a single test case against generated code"""
    if function_name:
        # Functional testing - call the function directly
        return run_functional_test(code, test_case, function_name, timeout)
    else:
        # Stdin/stdout testing - run as script
        return run_stdio_test(code, test_case, timeout)


def run_stdio_test(code: str, test_case: TestCase, timeout: int = 10) -> Tuple[bool, Optional[str]]:
    """Run stdin/stdout test using subprocess"""
    try:
        # Create a temporary file with the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Run the code with the input and capture output
            process = subprocess.run(
                [sys.executable, temp_file],
                input=test_case.input_data,
                text=True,
                capture_output=True,
                timeout=timeout
            )
            
            if process.returncode != 0:
                return False, f"Runtime error: {process.stderr}"
            
            # Compare output
            actual_output = process.stdout.strip()
            expected_output = test_case.expected_output.strip()
            
            if actual_output == expected_output:
                return True, None
            else:
                return False, f"Expected: {expected_output}, Got: {actual_output}"
                
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass
                
    except subprocess.TimeoutExpired:
        return False, "Time limit exceeded"
    except Exception as e:
        return False, str(e)


def run_functional_test(code: str, test_case: TestCase, function_name: str, timeout: int = 10) -> Tuple[bool, Optional[str]]:
    """Run functional test by calling the function directly"""
    try:
        import json as json_module
        
        # Parse input as JSON
        try:
            input_data = json_module.loads(test_case.input_data)
            expected_output = json_module.loads(test_case.expected_output)
        except json_module.JSONDecodeError as e:
            return False, f"JSON parse error: {e}"
        
        # For functional tests, the input_data should be treated as a single argument
        # (the function expects one argument which is the parsed JSON data)
        input_args = [input_data]
        
        # Create a safe execution environment
        exec_globals = {
            '__builtins__': __builtins__,
            'List': list,  # Common type hint
            'Dict': dict,
            'Set': set,
            'Tuple': tuple,
            'Optional': type(None),
            'Union': type(None),
        }
        
        # Execute the code
        exec(code, exec_globals)
        
        # Find the function or method
        if 'Solution' in exec_globals:
            # LeetCode-style class
            solution_instance = exec_globals['Solution']()
            if hasattr(solution_instance, function_name):
                func = getattr(solution_instance, function_name)
            else:
                return False, f"Method {function_name} not found in Solution class"
        elif function_name in exec_globals:
            # Standalone function
            func = exec_globals[function_name]
        else:
            return False, f"Function {function_name} not found"
        
        # Call the function with timeout
        def timeout_handler(signum, frame):
            raise TimeoutError("Function call timed out")
        
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            result = func(*input_args)
            signal.alarm(0)
        except TimeoutError:
            return False, "Time limit exceeded"
        except Exception as e:
            signal.alarm(0)
            return False, f"Runtime error: {e}"
        finally:
            signal.signal(signal.SIGALRM, old_handler)
        
        # Compare result with expected output
        if result == expected_output:
            return True, None
        else:
            return False, f"Expected: {expected_output}, Got: {result}"
            
    except Exception as e:
        return False, str(e)


def test_generated_code(problems: Dict[str, Problem], 
                       generated_code: Dict[str, str], 
                       timeout: int = 10) -> List[TestResult]:
    """Test all generated code against their respective test cases"""
    results = []
    
    # Only test problems that have generated code
    for task_id in generated_code.keys():
        if task_id not in problems:
            continue
            
        problem = problems[task_id]
        code = generated_code[task_id]
        
        # Fix function name if needed
        if problem.function_name:
            code = fix_function_name(code, problem.function_name)
        
        # Test against all test cases
        all_passed = True
        error_messages = []
        
        for i, test_case in enumerate(problem.test_cases):
            passed, error_msg = run_single_test(code, test_case, problem.function_name, timeout)
            
            if not passed:
                all_passed = False
                if error_msg:
                    error_messages.append(f"Test {i+1}: {error_msg}")
                else:
                    error_messages.append(f"Test {i+1}: Output mismatch")
        
        results.append(TestResult(
            task_id=task_id,
            passed=all_passed,
            error_message='; '.join(error_messages) if error_messages else None
        ))
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run tests on generated code')
    parser.add_argument('--test_file', default='livecode_test.jsonl',
                       help='Path to test cases JSONL file')
    parser.add_argument('--generation_file', required=True,
                       help='Path to generated code JSONL file')
    parser.add_argument('--output_file', 
                       help='Path to save results (optional)')
    parser.add_argument('--timeout', type=int, default=10,
                       help='Timeout for each test in seconds')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed results')
    
    args = parser.parse_args()
    
    # Load test problems
    print(f"Loading test problems from {args.test_file}...")
    problems = load_test_problems(args.test_file)
    print(f"Loaded {len(problems)} problems")
    
    # Load generated code
    print(f"Loading generated code from {args.generation_file}...")
    generated_code = load_generated_code(args.generation_file)
    print(f"Loaded code for {len(generated_code)} problems")
    
    # Run tests
    print("Running tests...")
    results = test_generated_code(problems, generated_code, args.timeout)
    
    # Calculate statistics
    total_problems = len(results)
    passed_problems = sum(1 for r in results if r.passed)
    pass_rate = (passed_problems / total_problems) * 100 if total_problems > 0 else 0
    
    print(f"\nResults:")
    print(f"Total problems: {total_problems}")
    print(f"Passed: {passed_problems}")
    print(f"Failed: {total_problems - passed_problems}")
    print(f"Pass rate: {pass_rate:.2f}%")
    
    if args.verbose:
        print("\nDetailed results:")
        for result in results:
            status = "PASS" if result.passed else "FAIL"
            print(f"{result.task_id}: {status}")
            if not result.passed and result.error_message:
                print(f"  Error: {result.error_message}")
    
    # Save results if requested
    if args.output_file:
        with open("results/"+args.generation_file.split("/")[1], 'w') as f:
            for result in results:
                output_line = {
                    'task_id': result.task_id,
                    'test_result': 1 if result.passed else 0
                }
                f.write(json.dumps(output_line) + '\n')
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()