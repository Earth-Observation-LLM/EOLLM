"""VQA quality validation.

This module provides comprehensive validation of generated VQA examples,
ensuring quality, diversity, and adherence to requirements.
"""

from typing import Dict, Any, List, Set
from collections import Counter


class VQAValidator:
    """Validator for VQA example quality and diversity.
    
    Validates:
    - Question count within specified range
    - Answer format (Yes/No/No clue)
    - Category distribution
    - No duplicate questions
    - Non-empty reasoning
    - Overall quality metrics
    
    Attributes:
        min_questions: Minimum number of questions
        max_questions: Maximum number of questions
        categories: Expected categories
        min_per_category: Minimum questions per category
    """
    
    def __init__(
        self,
        min_questions: int = 15,
        max_questions: int = 20,
        categories: List[str] = None,
        min_per_category: int = 2
    ):
        """Initialize VQA validator.
        
        Args:
            min_questions: Minimum number of questions
            max_questions: Maximum number of questions
            categories: Expected category list
            min_per_category: Minimum questions per category
        """
        self.min_questions = min_questions
        self.max_questions = max_questions
        self.categories = categories or [
            "sustainability",
            "infrastructure",
            "economic",
            "maintenance",
            "development"
        ]
        self.min_per_category = min_per_category
    
    def validate(self, vqa_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate VQA data comprehensively.
        
        Args:
            vqa_data: VQA data dictionary with 'vqa_examples' key
            
        Returns:
            Validation report with is_valid, warnings, errors, and statistics
        """
        report = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {},
            'category_distribution': {}
        }
        
        # Extract examples
        if 'vqa_examples' not in vqa_data:
            report['is_valid'] = False
            report['errors'].append("Missing 'vqa_examples' key in VQA data")
            return report
        
        examples = vqa_data['vqa_examples']
        
        if not isinstance(examples, list):
            report['is_valid'] = False
            report['errors'].append("'vqa_examples' must be a list")
            return report
        
        # Run validation checks
        self._validate_question_count(examples, report)
        self._validate_answer_formats(examples, report)
        self._validate_categories(examples, report)
        self._check_duplicates(examples, report)
        self._validate_reasoning(examples, report)
        self._compute_statistics(examples, report)
        
        # Set overall validity
        report['is_valid'] = len(report['errors']) == 0
        
        return report
    
    def _validate_question_count(
        self,
        examples: List[Dict],
        report: Dict[str, Any]
    ) -> None:
        """Validate question count is within range.
        
        Args:
            examples: List of VQA examples
            report: Validation report to update
        """
        count = len(examples)
        
        if count == 0:
            report['errors'].append("No questions generated")
        elif count < self.min_questions:
            report['warnings'].append(
                f"Only {count} questions generated (minimum: {self.min_questions})"
            )
        elif count > self.max_questions:
            report['warnings'].append(
                f"{count} questions generated (maximum: {self.max_questions})"
            )
    
    def _validate_answer_formats(
        self,
        examples: List[Dict],
        report: Dict[str, Any]
    ) -> None:
        """Validate answer formats are valid.
        
        Args:
            examples: List of VQA examples
            report: Validation report to update
        """
        valid_answers = {'Yes', 'No', 'No clue'}
        invalid_count = 0
        
        for i, example in enumerate(examples):
            if 'answer' not in example:
                report['errors'].append(f"Question {i+1} missing 'answer' field")
                invalid_count += 1
                continue
            
            answer = example['answer']
            if answer not in valid_answers:
                report['errors'].append(
                    f"Question {i+1} has invalid answer: '{answer}' "
                    f"(must be one of {valid_answers})"
                )
                invalid_count += 1
        
        if invalid_count > 0:
            report['statistics']['invalid_answers'] = invalid_count
    
    def _validate_categories(
        self,
        examples: List[Dict],
        report: Dict[str, Any]
    ) -> None:
        """Validate category distribution.
        
        Args:
            examples: List of VQA examples
            report: Validation report to update
        """
        # Count by category
        category_counts = Counter()
        uncategorized = 0
        
        for i, example in enumerate(examples):
            if 'category' not in example:
                report['warnings'].append(
                    f"Question {i+1} missing 'category' field"
                )
                uncategorized += 1
                continue
            
            category = example['category'].lower()
            category_counts[category] += 1
        
        # Store distribution
        report['category_distribution'] = dict(category_counts)
        
        if uncategorized > 0:
            report['statistics']['uncategorized'] = uncategorized
        
        # Check minimum per category
        for category in self.categories:
            count = category_counts.get(category, 0)
            if count < self.min_per_category:
                report['warnings'].append(
                    f"Category '{category}' has only {count} question(s) "
                    f"(minimum: {self.min_per_category})"
                )
        
        # Check for unexpected categories
        for category in category_counts:
            if category not in self.categories:
                report['warnings'].append(
                    f"Unexpected category: '{category}'"
                )
    
    def _check_duplicates(
        self,
        examples: List[Dict],
        report: Dict[str, Any]
    ) -> None:
        """Check for duplicate questions.
        
        Args:
            examples: List of VQA examples
            report: Validation report to update
        """
        questions: Set[str] = set()
        duplicates = []
        
        for i, example in enumerate(examples):
            if 'question' not in example:
                report['errors'].append(f"Question {i+1} missing 'question' field")
                continue
            
            question = example['question'].strip().lower()
            
            if question in questions:
                duplicates.append((i+1, example['question']))
            else:
                questions.add(question)
        
        if duplicates:
            report['warnings'].append(
                f"Found {len(duplicates)} duplicate question(s)"
            )
            report['statistics']['duplicate_questions'] = len(duplicates)
    
    def _validate_reasoning(
        self,
        examples: List[Dict],
        report: Dict[str, Any]
    ) -> None:
        """Validate reasoning fields are present and non-empty.
        
        Args:
            examples: List of VQA examples
            report: Validation report to update
        """
        missing_reasoning = 0
        empty_reasoning = 0
        
        for i, example in enumerate(examples):
            if 'reasoning' not in example:
                missing_reasoning += 1
            elif not example['reasoning'] or not example['reasoning'].strip():
                empty_reasoning += 1
        
        if missing_reasoning > 0:
            report['warnings'].append(
                f"{missing_reasoning} question(s) missing 'reasoning' field"
            )
        
        if empty_reasoning > 0:
            report['warnings'].append(
                f"{empty_reasoning} question(s) have empty reasoning"
            )
        
        total_issues = missing_reasoning + empty_reasoning
        if total_issues > 0:
            report['statistics']['reasoning_issues'] = total_issues
    
    def _compute_statistics(
        self,
        examples: List[Dict],
        report: Dict[str, Any]
    ) -> None:
        """Compute overall statistics.
        
        Args:
            examples: List of VQA examples
            report: Validation report to update
        """
        stats = report['statistics']
        
        # Basic counts
        stats['total_questions'] = len(examples)
        
        # Answer distribution
        answer_counts = Counter(
            ex.get('answer', 'missing') for ex in examples
        )
        stats['answer_distribution'] = dict(answer_counts)
        
        # Average question length
        question_lengths = [
            len(ex.get('question', ''))
            for ex in examples
        ]
        if question_lengths:
            stats['avg_question_length'] = sum(question_lengths) / len(question_lengths)
        
        # Average reasoning length
        reasoning_lengths = [
            len(ex.get('reasoning', ''))
            for ex in examples
        ]
        if reasoning_lengths:
            stats['avg_reasoning_length'] = sum(reasoning_lengths) / len(reasoning_lengths)
    
    def is_valid(self, vqa_data: Dict[str, Any]) -> bool:
        """Quick validation check.
        
        Args:
            vqa_data: VQA data to validate
            
        Returns:
            True if valid, False otherwise
        """
        report = self.validate(vqa_data)
        return report['is_valid']
    
    def get_quality_score(self, vqa_data: Dict[str, Any]) -> float:
        """Compute a quality score (0.0 to 1.0).
        
        Args:
            vqa_data: VQA data to score
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        report = self.validate(vqa_data)
        
        # Start with 1.0
        score = 1.0
        
        # Deduct for errors (more severe)
        score -= len(report['errors']) * 0.1
        
        # Deduct for warnings (less severe)
        score -= len(report['warnings']) * 0.05
        
        # Ensure in range
        return max(0.0, min(1.0, score))

