"""
LiveCodeBench Dataset Parser
Similar to BigCodeBenchMutation parser but for LiveCodeBench code generation dataset.
"""

import json
import zlib
import pickle
import base64
from enum import Enum
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datasets import load_dataset


class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)


@dataclass
class LiveCodeBenchProblem:
    """LiveCodeBench problem with assigned ID"""
    id: str  # Assigned unique ID
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: List[Test]
    private_test_cases: List[Test]
    metadata: Dict[str, Any]

    def __post_init__(self):
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        self.contest_date = datetime.fromisoformat(self.contest_date)

        # Parse test cases
        if isinstance(self.public_test_cases, str):
            self.public_test_cases = json.loads(self.public_test_cases)
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]

        # Handle compressed private test cases
        if isinstance(self.private_test_cases, str):
            try:
                self.private_test_cases = json.loads(self.private_test_cases)
            except:
                self.private_test_cases = json.loads(
                    pickle.loads(
                        zlib.decompress(
                            base64.b64decode(self.private_test_cases.encode("utf-8"))
                        )
                    )
                )
        self.private_test_cases = [Test(**t) for t in self.private_test_cases]

        # Parse metadata
        if isinstance(self.metadata, str):
            self.metadata = json.loads(self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "id": self.id,
            "question_title": self.question_title,
            "question_content": self.question_content,
            "platform": self.platform.value,
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat(),
            "starter_code": self.starter_code,
            "difficulty": self.difficulty.value,
            "public_test_cases": [
                {
                    "input": t.input,
                    "output": t.output,
                    "testtype": t.testtype.value
                }
                for t in self.public_test_cases
            ],
            "private_test_cases": [
                {
                    "input": t.input,
                    "output": t.output,
                    "testtype": t.testtype.value
                }
                for t in self.private_test_cases
            ],
            "metadata": self.metadata,
        }

    def get_function_name(self) -> Optional[str]:
        """Extract function name from metadata"""
        return self.metadata.get("func_name", None)

    def get_test_type(self) -> TestType:
        """Determine the test type (stdin or functional)"""
        if self.public_test_cases:
            return self.public_test_cases[0].testtype
        elif self.private_test_cases:
            return self.private_test_cases[0].testtype
        return TestType.FUNCTIONAL

    def get_all_test_cases(self) -> List[Test]:
        """Get all test cases (public + private)"""
        return self.public_test_cases + self.private_test_cases


class LiveCodeBenchParser:
    """Parser for LiveCodeBench dataset"""
    
    def __init__(self, version_tag: str = "release_v6"):
        self.version_tag = version_tag
        self.problems: List[LiveCodeBenchProblem] = []
        
    def load_dataset(self, start_date: Optional[str] = None, end_date: Optional[str] = None, 
                     max_problems: Optional[int] = None, streaming: bool = True) -> List[LiveCodeBenchProblem]:
        """
        Load LiveCodeBench dataset and assign IDs to each instance
        
        Args:
            start_date: Filter problems after this date (YYYY-MM-DD)
            end_date: Filter problems before this date (YYYY-MM-DD)
            max_problems: Maximum number of problems to load (for testing)
            streaming: Use streaming mode to reduce memory usage
            
        Returns:
            List of LiveCodeBenchProblem with assigned IDs
        """
        print(f"Loading LiveCodeBench dataset {self.version_tag}...")
        
        try:
            # Try streaming mode first to reduce memory usage
            if streaming:
                dataset = load_dataset(
                    "livecodebench/code_generation_lite", 
                    split="test", 
                    version_tag=self.version_tag, 
                    trust_remote_code=True,
                    streaming=True
                )
                print("Using streaming mode to reduce memory usage...")
            else:
                dataset = load_dataset(
                    "livecodebench/code_generation_lite", 
                    split="test", 
                    version_tag=self.version_tag, 
                    trust_remote_code=True
                )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Trying without streaming...")
            dataset = load_dataset(
                "livecodebench/code_generation_lite", 
                split="test", 
                version_tag=self.version_tag, 
                trust_remote_code=True
            )
        
        # Parse date filters once
        p_start_date = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        p_end_date = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
        
        # Convert to LiveCodeBenchProblem objects and assign IDs
        problems = []
        processed_count = 0
        
        try:
            if streaming:
                # Process streaming dataset
                for idx, problem_data in enumerate(dataset):
                    if max_problems and idx >= max_problems:
                        break
                        
                    try:
                        # Assign unique ID: lcb_{version}_{index}
                        problem_id = f"lcb_{self.version_tag}_{idx:04d}"
                        
                        # Create problem object with assigned ID
                        problem = LiveCodeBenchProblem(
                            id=problem_id,
                            **problem_data
                        )
                        
                        # Apply date filtering during processing to save memory
                        if p_start_date and problem.contest_date < p_start_date:
                            continue
                        if p_end_date and problem.contest_date > p_end_date:
                            continue
                            
                        problems.append(problem)
                        processed_count += 1
                        
                        # Print progress periodically
                        if processed_count % 100 == 0:
                            print(f"Processed {processed_count} problems...")
                            import time
                            time.sleep(10)
                            
                    except Exception as e:
                        print(f"Error processing problem {idx}: {e}")
                        continue
            else:
                # Process regular dataset
                total_size = len(dataset)
                print(f"Loaded {total_size} raw problems")
                
                for idx, problem_data in enumerate(dataset):
                    if max_problems and idx >= max_problems:
                        break
                        
                    try:
                        # Assign unique ID: lcb_{version}_{index}
                        problem_id = f"lcb_{self.version_tag}_{idx:04d}"
                        
                        # Create problem object with assigned ID
                        problem = LiveCodeBenchProblem(
                            id=problem_id,
                            **problem_data
                        )
                        
                        # Apply date filtering
                        if p_start_date and problem.contest_date < p_start_date:
                            continue
                        if p_end_date and problem.contest_date > p_end_date:
                            continue
                            
                        problems.append(problem)
                        processed_count += 1
                        
                    except Exception as e:
                        print(f"Error processing problem {idx}: {e}")
                        continue
        
        except Exception as e:
            print(f"Error during dataset processing: {e}")
            return []
        
        self.problems = problems
        print(f"Successfully processed {len(problems)} problems")
        
        return problems
    
    def get_problem_by_id(self, problem_id: str) -> Optional[LiveCodeBenchProblem]:
        """Get problem by its assigned ID"""
        for problem in self.problems:
            if problem.id == problem_id:
                return problem
        return None
    
    def get_problems_by_platform(self, platform: Platform) -> List[LiveCodeBenchProblem]:
        """Get all problems from a specific platform"""
        return [p for p in self.problems if p.platform == platform]
    
    def get_problems_by_difficulty(self, difficulty: Difficulty) -> List[LiveCodeBenchProblem]:
        """Get all problems of a specific difficulty"""
        return [p for p in self.problems if p.difficulty == difficulty]
    
    def get_functional_problems(self) -> List[LiveCodeBenchProblem]:
        """Get problems that use functional testing (have starter code)"""
        return [p for p in self.problems if p.get_test_type() == TestType.FUNCTIONAL]
    
    def get_stdin_problems(self) -> List[LiveCodeBenchProblem]:
        """Get problems that use stdin/stdout testing"""
        return [p for p in self.problems if p.get_test_type() == TestType.STDIN]
    
    def export_to_json(self, output_file: str) -> None:
        """Export all problems to JSON file"""
        data = [problem.to_dict() for problem in self.problems]
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Exported {len(data)} problems to {output_file}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        stats = {
            "total_problems": len(self.problems),
            "platforms": {},
            "difficulties": {},
            "test_types": {},
            "date_range": {
                "earliest": None,
                "latest": None
            }
        }
        
        # Count by platform
        for platform in Platform:
            count = len(self.get_problems_by_platform(platform))
            stats["platforms"][platform.value] = count
        
        # Count by difficulty
        for difficulty in Difficulty:
            count = len(self.get_problems_by_difficulty(difficulty))
            stats["difficulties"][difficulty.value] = count
        
        # Count by test type
        functional_count = len(self.get_functional_problems())
        stdin_count = len(self.get_stdin_problems())
        stats["test_types"]["functional"] = functional_count
        stats["test_types"]["stdin"] = stdin_count
        
        # Date range
        if self.problems:
            dates = [p.contest_date for p in self.problems]
            stats["date_range"]["earliest"] = min(dates).isoformat()
            stats["date_range"]["latest"] = max(dates).isoformat()
        
        return stats


def load_full_dataset():
    """Load the full dataset (use with caution due to memory requirements)"""
    parser = LiveCodeBenchParser(version_tag="release_v6")
    
    print("Loading full dataset...")
    problems = parser.load_dataset(streaming=True)  # No max_problems limit
    
    if problems:
        print(f"Successfully loaded {len(problems)} problems")
        parser.export_to_json("livecodebench_full_dataset.json")
    else:
        print("Failed to load dataset")
    
    return problems

def main():
    """Example usage"""
    parser = LiveCodeBenchParser(version_tag="release_v6")
    
    # Load dataset with memory optimization
    print("Loading dataset with streaming mode and limited size for testing...")
    problems = parser.load_full_dataset() # Load only 50 for testing
    
    if not problems:
        print("Failed to load any problems")
        return
    
    # Print statistics
    stats = parser.get_statistics()
    print("\nDataset Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Example: Get first few problems
    print(f"\nFirst 3 problem IDs:")
    for i in range(min(3, len(problems))):
        problem = problems[i]
        print(f"- {problem.id}: {problem.question_title} ({problem.platform.value}, {problem.difficulty.value})")
        print(f"  Test type: {problem.get_test_type().value}")
        print(f"  Public tests: {len(problem.public_test_cases)}, Private tests: {len(problem.private_test_cases)}")
        print()
    
    # Example: Export to JSON (uncomment to use)
    # parser.export_to_json("livecodebench_problems_sample.json")
    
    print(f"Successfully processed {len(problems)} problems!")



if __name__ == "__main__":
    load_full_dataset()