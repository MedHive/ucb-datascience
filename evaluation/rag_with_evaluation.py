#!/usr/bin/env python3
"""
RAG System with Real-time Hallucination Evaluation
Wrapper that adds evaluation capabilities to RAG systems
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from .llm_judge import HallucinationJudge
    from .config import HALLUCINATION_THRESHOLDS
except ImportError:
    from llm_judge import HallucinationJudge
    from config import HALLUCINATION_THRESHOLDS


class EvaluatedRAGWrapper:
    """
    Wrapper class that adds real-time hallucination evaluation to any RAG system
    """

    def __init__(self, rag_system, enable_evaluation: bool = True):
        """
        Initialize the evaluated RAG wrapper.

        Args:
            rag_system: The underlying RAG system to wrap
            enable_evaluation: Whether to enable real-time evaluation
        """
        self.rag_system = rag_system
        self.enable_evaluation = enable_evaluation
        self.judge = HallucinationJudge() if enable_evaluation else None
        self.evaluation_history = []

    def query(self, question: str, **kwargs):
        """
        Execute RAG query with optional real-time evaluation.

        Args:
            question: User question
            **kwargs: Additional arguments passed to underlying RAG system

        Returns:
            Dict with response and optional evaluation
        """
        # Execute the underlying RAG query
        result = self.rag_system.query(question, **kwargs)

        # If evaluation is disabled, return as-is
        if not self.enable_evaluation or not self.judge:
            return result

        # Extract answer and context
        answer = result.get("answer", "")
        context = result.get("context", [])

        # Format context for evaluation
        # Handle different context formats (list of dicts vs list of strings)
        if context and isinstance(context, list):
            if isinstance(context[0], dict):
                # GraphRAG format: list of dicts with 'text' field
                context_text = "\n\n".join([
                    c.get("text", "") for c in context if c.get("text")
                ])
            else:
                # Simple format: list of strings
                context_text = "\n\n".join(context)
        else:
            context_text = str(context) if context else ""

        # Evaluate for hallucinations
        print("\n🔍 Evaluating response for hallucinations...")
        evaluation = self.judge.evaluate(
            query=question,
            response=answer,
            context=context_text,
            parse_inline_context=True
        )

        # Store evaluation history
        self.evaluation_history.append({
            "question": question,
            "answer": answer,
            "evaluation": evaluation
        })

        # Add evaluation to result
        result["evaluation"] = evaluation

        # Display evaluation summary
        self._display_evaluation_summary(evaluation)

        return result

    def _display_evaluation_summary(self, evaluation: dict):
        """Display a formatted evaluation summary"""
        score = evaluation.get("hallucination_score", 0.0)
        severity = evaluation.get("severity", "unknown")
        explanation = evaluation.get("explanation", "No explanation provided")

        print("\n" + "="*80)
        print("📊 HALLUCINATION EVALUATION")
        print("="*80)

        # Color-coded severity display
        severity_emoji = {
            "none": "✅",
            "minor": "⚠️",
            "moderate": "🟠",
            "severe": "🔴",
            "error": "❌"
        }

        emoji = severity_emoji.get(severity, "❓")
        print(f"{emoji} Hallucination Score: {score:.3f} / 1.0")
        print(f"   Severity: {severity.upper()}")
        print(f"\n💬 Explanation:")
        print(f"   {explanation}")

        # Display claims analysis
        supported = evaluation.get("supported_claims", [])
        unsupported = evaluation.get("unsupported_claims", [])

        if supported:
            print(f"\n✓ Supported Claims ({len(supported)}):")
            for claim in supported[:3]:  # Show top 3
                print(f"   • {claim}")
            if len(supported) > 3:
                print(f"   ... and {len(supported) - 3} more")

        if unsupported:
            print(f"\n✗ Unsupported Claims ({len(unsupported)}):")
            for claim in unsupported:
                print(f"   • {claim}")

        print("="*80)

    def get_evaluation_summary(self):
        """Get summary of all evaluations in this session"""
        if not self.evaluation_history:
            return {"message": "No evaluations performed yet"}

        scores = [e["evaluation"]["hallucination_score"] for e in self.evaluation_history
                 if e["evaluation"].get("hallucination_score") is not None]

        return {
            "total_queries": len(self.evaluation_history),
            "mean_hallucination_score": sum(scores) / len(scores) if scores else None,
            "min_score": min(scores) if scores else None,
            "max_score": max(scores) if scores else None,
            "severity_counts": {
                severity: sum(1 for e in self.evaluation_history
                            if e["evaluation"]["severity"] == severity)
                for severity in ["none", "minor", "moderate", "severe", "error"]
            }
        }


def demonstrate_evaluation():
    """
    Demonstration function showing how to use the evaluation wrapper
    """
    print("="*80)
    print("RAG System with Real-time Hallucination Evaluation - Demo")
    print("="*80)

    # This is a demo - in practice, you would wrap your actual RAG system
    print("\nThis wrapper can be used with any RAG system:")
    print("  1. DocumentRAG (testRAG.py)")
    print("  2. GraphRAG (testMDKGRAG.py)")
    print("\nExample usage:")
    print("""
    # Import your RAG system
    from DocumentRAG.testRAG import rag_query

    # Or for GraphRAG
    from DocumentRAG.testMDKGRAG import ImprovedGraphRAG

    # Wrap with evaluation
    evaluated_rag = EvaluatedRAGWrapper(rag_system, enable_evaluation=True)

    # Query with automatic evaluation
    result = evaluated_rag.query("What is Capnostream20?")

    # Access evaluation
    eval_score = result['evaluation']['hallucination_score']
    print(f"Hallucination score: {eval_score}")
    """)

    print("\n" + "="*80)
    print("To integrate with your RAG systems:")
    print("  • Both testRAG.py and testMDKGRAG.py now include context tags")
    print("  • Pass enable_evaluation=True to enable real-time checking")
    print("  • Evaluation results are displayed after each query")
    print("="*80)


if __name__ == "__main__":
    demonstrate_evaluation()
