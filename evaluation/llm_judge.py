"""
LLM-as-a-Judge Hallucination Evaluator
Uses GPT-4o-mini to evaluate hallucinations in RAG system responses
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from datetime import datetime

try:
    from .config import (
        JUDGE_PROMPT_TEMPLATE,
        get_judge_config,
        HALLUCINATION_THRESHOLDS
    )
except ImportError:
    from config import (
        JUDGE_PROMPT_TEMPLATE,
        get_judge_config,
        HALLUCINATION_THRESHOLDS
    )


class HallucinationJudge:
    """
    LLM-based hallucination detector for RAG systems.

    Evaluates whether generated responses are grounded in retrieved context,
    using an LLM as a judge to assign hallucination scores (0.0 to 1.0).
    """

    def __init__(self, model: str = None, temperature: float = None):
        """
        Initialize the hallucination judge.

        Args:
            model: LLM model to use (default: from config)
            temperature: Sampling temperature (default: from config)
        """
        config = get_judge_config()
        self.model = model or config["model"]
        self.temperature = temperature or config["temperature"]
        self.client = OpenAI(api_key=config["api_key"])
        self.evaluation_history = []

    def parse_context_tags(self, response: str) -> Tuple[str, str]:
        """
        Parse inline <Context> tags from RAG response.

        Args:
            response: RAG response with inline context tags

        Returns:
            Tuple of (clean_response, extracted_context)
        """
        # Extract context from <Context>...</Context> tags
        context_pattern = r'<Context>(.*?)</Context>'
        contexts = re.findall(context_pattern, response, re.DOTALL | re.IGNORECASE)

        # Remove context tags from response
        clean_response = re.sub(context_pattern, '', response, flags=re.DOTALL | re.IGNORECASE)
        clean_response = clean_response.strip()

        # Combine all extracted contexts
        extracted_context = "\n\n".join(contexts) if contexts else ""

        return clean_response, extracted_context

    def evaluate(
        self,
        query: str,
        response: str,
        context: Optional[str] = None,
        parse_inline_context: bool = True
    ) -> Dict:
        """
        Evaluate a RAG response for hallucinations.

        Args:
            query: User's original query
            response: Generated response (may include <Context> tags)
            context: Retrieved context (optional if inline context present)
            parse_inline_context: Whether to parse <Context> tags from response

        Returns:
            Dict with evaluation results:
            {
                "hallucination_score": float,
                "explanation": str,
                "unsupported_claims": List[str],
                "supported_claims": List[str],
                "severity": str,
                "timestamp": str
            }
        """
        # Parse inline context if enabled
        if parse_inline_context:
            clean_response, inline_context = self.parse_context_tags(response)
            # Use inline context if no explicit context provided
            if not context and inline_context:
                context = inline_context
            response = clean_response

        # Validate inputs
        if not context:
            raise ValueError("No context provided. Either pass context explicitly or enable parse_inline_context.")

        # Format prompt
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            query=query,
            context=context,
            response=response
        )

        # Call LLM judge
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for AI system outputs."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )

            # Parse JSON response
            result = json.loads(completion.choices[0].message.content)

            # Add metadata
            result["severity"] = self._classify_severity(result["hallucination_score"])
            result["timestamp"] = datetime.now().isoformat()
            result["query"] = query
            result["model"] = self.model

            # Store in history
            self.evaluation_history.append(result)

            return result

        except json.JSONDecodeError as e:
            # Fallback if JSON parsing fails
            return {
                "hallucination_score": None,
                "explanation": f"Failed to parse judge response: {str(e)}",
                "unsupported_claims": [],
                "supported_claims": [],
                "severity": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        except Exception as e:
            return {
                "hallucination_score": None,
                "explanation": f"Evaluation failed: {str(e)}",
                "unsupported_claims": [],
                "supported_claims": [],
                "severity": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def _classify_severity(self, score: float) -> str:
        """Classify hallucination severity based on score"""
        if score is None:
            return "error"
        elif score <= HALLUCINATION_THRESHOLDS["no_hallucination"]:
            return "none"
        elif score <= HALLUCINATION_THRESHOLDS["minor_hallucination"]:
            return "minor"
        elif score <= HALLUCINATION_THRESHOLDS["moderate_hallucination"]:
            return "moderate"
        else:
            return "severe"

    def batch_evaluate(
        self,
        test_cases: List[Dict],
        verbose: bool = True
    ) -> List[Dict]:
        """
        Evaluate multiple test cases in batch.

        Args:
            test_cases: List of dicts with keys: query, response, context
            verbose: Print progress

        Returns:
            List of evaluation results
        """
        results = []

        for i, test_case in enumerate(test_cases):
            if verbose:
                print(f"Evaluating test case {i+1}/{len(test_cases)}: {test_case.get('query', 'Unknown')[:50]}...")

            result = self.evaluate(
                query=test_case["query"],
                response=test_case["response"],
                context=test_case.get("context"),
                parse_inline_context=test_case.get("parse_inline_context", True)
            )

            # Add test case metadata
            result["test_case_id"] = test_case.get("id", i)
            result["category"] = test_case.get("category", "unknown")
            if "expected_score" in test_case:
                result["expected_score"] = test_case["expected_score"]
                result["score_error"] = abs(result["hallucination_score"] - test_case["expected_score"])

            results.append(result)

            if verbose:
                print(f"  Score: {result['hallucination_score']:.2f} ({result['severity']})")

        return results

    def summarize_results(self, results: List[Dict]) -> Dict:
        """
        Generate summary statistics for evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            Dict with summary statistics
        """
        if not results:
            return {"error": "No results to summarize"}

        scores = [r["hallucination_score"] for r in results if r["hallucination_score"] is not None]

        summary = {
            "total_cases": len(results),
            "successful_evaluations": len(scores),
            "failed_evaluations": len(results) - len(scores),
            "mean_score": sum(scores) / len(scores) if scores else None,
            "min_score": min(scores) if scores else None,
            "max_score": max(scores) if scores else None,
            "severity_distribution": {
                "none": sum(1 for r in results if r["severity"] == "none"),
                "minor": sum(1 for r in results if r["severity"] == "minor"),
                "moderate": sum(1 for r in results if r["severity"] == "moderate"),
                "severe": sum(1 for r in results if r["severity"] == "severe"),
                "error": sum(1 for r in results if r["severity"] == "error")
            }
        }

        # Calculate accuracy if expected scores provided
        if any("expected_score" in r for r in results):
            errors = [r["score_error"] for r in results if "score_error" in r]
            summary["mean_absolute_error"] = sum(errors) / len(errors) if errors else None

        return summary


def quick_evaluate(query: str, response: str, context: str = None) -> float:
    """
    Quick evaluation function for single responses.

    Args:
        query: User query
        response: RAG response (with or without <Context> tags)
        context: Retrieved context (optional)

    Returns:
        Hallucination score (0.0 to 1.0)
    """
    judge = HallucinationJudge()
    result = judge.evaluate(query, response, context)
    return result["hallucination_score"]
