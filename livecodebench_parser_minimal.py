"""
Minimal LiveCodeBench Parser - Keeps essential fields plus date and test cases
Removes only unnecessary metadata to reduce memory usage
"""

import json
import zlib
import pickle
import base64
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datasets import load_dataset


@dataclass
class TestCase:
    input: str
    output: str
    testtype: str


@dataclass
class MinimalLiveCodeBenchProblem:
    """Minimal LiveCodeBench problem with essential fields, date, and test cases"""
    id: str
    question_title: str
    question_content: str
    platform: str
    difficulty: str
    starter_code: str
    contest_date: str  # Keep as ISO string to save memory
    public_test_cases: List[TestCase]
    private_test_cases: List[TestCase]
    
    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        return {
            "id": self.id,
            "question_title": self.question_title,
            "question_content": self.question_content,
            "platform": self.platform,
            "difficulty": self.difficulty,
            "starter_code": self.starter_code,
            "contest_date": self.contest_date,
            "public_test_cases": [
                {
                    "input": t.input,
                    "output": t.output,
                    "testtype": t.testtype
                }
                for t in self.public_test_cases
            ],
            "private_test_cases": [
                {
                    "input": t.input,
                    "output": t.output,
                    "testtype": t.testtype
                }
                for t in self.private_test_cases
            ]
        }
    
    def format_for_mutation(self) -> str:
        """Format as question for mutation (same as BigCodeBench format)"""
        question = f"Problem: {self.question_title}\n\n{self.question_content}"
        if self.starter_code.strip():
            question += f"\n\nStarter Code:\n{self.starter_code}"
        return question


class MinimalLiveCodeBenchParser:
    """Minimal parser that extracts essential fields, date, and test cases"""
    
    def __init__(self, version_tag: str = "release_v6"):
        self.version_tag = version_tag
    
    def _parse_test_cases(self, test_cases_data):
        """Parse test cases from dataset format"""
        if isinstance(test_cases_data, str):
            try:
                test_cases_data = json.loads(test_cases_data)
            except:
                # Handle compressed test cases
                try:
                    test_cases_data = json.loads(
                        pickle.loads(
                            zlib.decompress(
                                base64.b64decode(test_cases_data.encode("utf-8"))
                            )
                        )
                    )
                except:
                    return []
        
        test_cases = []
        for t in test_cases_data:
            test_case = TestCase(
                input=str(t.get('input', '')),
                output=str(t.get('output', '')),
                testtype=str(t.get('testtype', 'functional'))
            )
            test_cases.append(test_case)
        
        return test_cases
    
    def load_dataset_minimal(self, max_problems: Optional[int] = None, streaming: bool = True) -> List[MinimalLiveCodeBenchProblem]:
        """
        Load LiveCodeBench dataset with essential fields, date, and test cases
        
        Args:
            max_problems: Maximum number of problems to load (None for all)
            streaming: Use streaming mode to reduce memory usage
            
        Returns:
            List of MinimalLiveCodeBenchProblem with minimal but complete data
        """
        print(f"Loading minimal LiveCodeBench dataset {self.version_tag}...")
        
        try:
            dataset = load_dataset(
                "livecodebench/code_generation_lite", 
                split="test", 
                version_tag=self.version_tag, 
                trust_remote_code=True,
                streaming=streaming
            )
            print("Using streaming mode to reduce memory usage...")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []
        
        problems = []
        processed_count = 0
        
        try:
            for idx, problem_data in enumerate(dataset):
                if max_problems and idx >= max_problems:
                    break
                
                try:
                    # Assign unique ID
                    problem_id = f"lcb_{self.version_tag}_{idx:04d}"
                    
                    # Parse contest date
                    contest_date = problem_data.get('contest_date', '')
                    if contest_date:
                        try:
                            # Convert to datetime and back to ISO string to normalize
                            dt = datetime.fromisoformat(contest_date)
                            contest_date = dt.isoformat()
                        except:
                            contest_date = str(contest_date)
                    
                    # Parse test cases
                    public_tests = self._parse_test_cases(problem_data.get('public_test_cases', []))
                    private_tests = self._parse_test_cases(problem_data.get('private_test_cases', []))
                    
                    # Create minimal problem object
                    problem = MinimalLiveCodeBenchProblem(
                        id=problem_id,
                        question_title=problem_data.get('question_title', ''),
                        question_content=problem_data.get('question_content', ''),
                        platform=problem_data.get('platform', ''),
                        difficulty=problem_data.get('difficulty', ''),
                        starter_code=problem_data.get('starter_code', ''),
                        contest_date=contest_date,
                        public_test_cases=public_tests,
                        private_test_cases=private_tests
                    )
                    
                    problems.append(problem)
                    processed_count += 1
                    
                    # Print progress periodically
                    if processed_count % 100 == 0:
                        print(f"Processed {processed_count} problems...")
                        
                except Exception as e:
                    print(f"Error processing problem {idx}: {e}")
                    continue
        
        except Exception as e:
            print(f"Error during dataset processing: {e}")
            return []
        
        print(f"Successfully processed {len(problems)} minimal problems")
        return problems
    
    def save_minimal_dataset(self, problems: List[MinimalLiveCodeBenchProblem], output_file: str):
        """Save minimal dataset to JSON file"""
        data = [problem.to_dict() for problem in problems]
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(data)} minimal problems to {output_file}")
    
    def save_simple_jsonl(self, problems: List[MinimalLiveCodeBenchProblem], output_file: str):
        """Save dataset to JSONL file with only id, question, starter_code, and contest_date"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for problem in problems:
                # Combine question_title and question_content into a single question field
                question = f"{problem.question_title}\n\n{problem.question_content}"
                
                simple_data = {
                    "id": problem.id,
                    "question": question,
                    "starter_code": problem.starter_code,
                    "contest_date": problem.contest_date
                }
                
                f.write(json.dumps(simple_data, ensure_ascii=False) + '\n')
        print(f"Saved {len(problems)} problems to JSONL file: {output_file}")
    
    def load_from_minimal_json(self, json_file: str) -> List[MinimalLiveCodeBenchProblem]:
        """Load from minimal JSON file"""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        problems = []
        for item in data:
            # Parse test cases
            public_tests = [TestCase(**t) for t in item.get('public_test_cases', [])]
            private_tests = [TestCase(**t) for t in item.get('private_test_cases', [])]
            
            problem = MinimalLiveCodeBenchProblem(
                id=item['id'],
                question_title=item['question_title'],
                question_content=item['question_content'],
                platform=item['platform'],
                difficulty=item['difficulty'],
                starter_code=item['starter_code'],
                contest_date=item['contest_date'],
                public_test_cases=public_tests,
                private_test_cases=private_tests
            )
            problems.append(problem)
        
        return problems


def create_minimal_dataset(output_file: str = "livecodebench_minimal.json", max_problems: Optional[int] = None):
    """Create a minimal dataset file with essential fields, date, and test cases"""
    parser = MinimalLiveCodeBenchParser(version_tag="release_v6")
    
    # Load minimal dataset
    problems = parser.load_dataset_minimal(max_problems=max_problems, streaming=True)
    
    if not problems:
        print("Failed to load any problems")
        return
    
    # Save minimal dataset
    parser.save_minimal_dataset(problems, output_file)
    
    # Save simple JSONL file with only id, question, and starter_code
    jsonl_file = output_file.replace('.json', '_simple.jsonl')
    parser.save_simple_jsonl(problems, jsonl_file)
    
    # Show size comparison
    import os
    if os.path.exists(output_file):
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"Minimal dataset size: {size_mb:.2f} MB")
    
    if os.path.exists(jsonl_file):
        size_mb = os.path.getsize(jsonl_file) / (1024 * 1024)
        print(f"Simple JSONL dataset size: {size_mb:.2f} MB")
    
    # Show what we kept
    print("\nFields kept in minimal dataset:")
    print("- id: Problem ID")
    print("- question_title: Problem title")
    print("- question_content: Problem description")
    print("- platform: Platform (leetcode/codeforces/atcoder)")
    print("- difficulty: Difficulty level")
    print("- starter_code: Function template (if any)")
    print("- contest_date: Contest date")
    print("- public_test_cases: Public test cases")
    print("- private_test_cases: Private test cases")
    print("\nFields in simple JSONL:")
    print("- id: Problem ID")
    print("- question: Combined title and content")
    print("- starter_code: Function template (if any)")
    print("- contest_date: Contest date")
    print("\nFields removed:")
    print("- metadata: Extra metadata")
    print("- contest_id: Contest information")
    print("- question_id: Original question ID")
    
    return problems


def demo_minimal_dataset():
    """Demo the minimal dataset"""
    print("Creating minimal LiveCodeBench dataset...")
    problems = create_minimal_dataset("livecodebench_minimal_demo.json", max_problems=50)
    
    if problems:
        print(f"\nFirst problem example:")
        print(f"ID: {problems[0].id}")
        print(f"Title: {problems[0].question_title}")
        print(f"Platform: {problems[0].platform}")
        print(f"Difficulty: {problems[0].difficulty}")
        print(f"Contest Date: {problems[0].contest_date}")
        print(f"Content length: {len(problems[0].question_content)} characters")
        print(f"Has starter code: {bool(problems[0].starter_code.strip())}")
        print(f"Public tests: {len(problems[0].public_test_cases)}")
        print(f"Private tests: {len(problems[0].private_test_cases)}")
        
        print(f"\nFormatted for mutation:")
        print("-" * 40)
        formatted = problems[0].format_for_mutation()
        print(formatted[:300] + "..." if len(formatted) > 300 else formatted)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create minimal LiveCodeBench dataset")
    parser.add_argument("--output", default="livecodebench_minimal.json", help="Output file name")
    parser.add_argument("--max_problems", type=int, help="Maximum problems to process")
    parser.add_argument("--demo", action="store_true", help="Run demo with small dataset")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_minimal_dataset()
    else:
        create_minimal_dataset(args.output, args.max_problems)