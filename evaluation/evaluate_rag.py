#!/usr/bin/env python3
"""
Standalone Batch Evaluation Script for RAG Systems
Evaluates hallucinations using LLM-as-a-Judge approach
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict

try:
    from .llm_judge import HallucinationJudge
    from .test_cases import get_test_cases, get_categories, print_test_case_summary
    from .config import RESULTS_DIR
except ImportError:
    from llm_judge import HallucinationJudge
    from test_cases import get_test_cases, get_categories, print_test_case_summary
    from config import RESULTS_DIR


def save_results(results: List[Dict], summary: Dict, output_path: str = None):
    """
    Save evaluation results to JSON file.

    Args:
        results: List of evaluation results
        summary: Summary statistics
        output_path: Output file path (default: auto-generated in results/)
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(RESULTS_DIR) / f"evaluation_results_{timestamp}.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_cases": len(results),
            "evaluator": "LLM-as-a-Judge (GPT-4o-mini)"
        },
        "summary": summary,
        "results": results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return str(output_path)


def print_results_table(results: List[Dict]):
    """Print results in a formatted table"""
    print("\n" + "="*100)
    print(f"{'ID':<12} {'Category':<12} {'Score':<8} {'Expected':<10} {'Severity':<12} {'Query':<40}")
    print("="*100)

    for r in results:
        test_id = r.get('test_case_id', 'unknown')
        category = r.get('category', 'unknown')
        score = r.get('hallucination_score', 0.0)
        expected = r.get('expected_score', 'N/A')
        severity = r.get('severity', 'unknown')
        query = r.get('query', '')[:37] + "..." if len(r.get('query', '')) > 40 else r.get('query', '')

        expected_str = f"{expected:.1f}" if isinstance(expected, (int, float)) else str(expected)
        print(f"{test_id:<12} {category:<12} {score:<8.2f} {expected_str:<10} {severity:<12} {query:<40}")

    print("="*100)


def print_summary(summary: Dict):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total cases:              {summary['total_cases']}")
    print(f"Successful evaluations:   {summary['successful_evaluations']}")
    print(f"Failed evaluations:       {summary['failed_evaluations']}")

    if summary.get('mean_score') is not None:
        print(f"\nHallucination Scores:")
        print(f"  Mean:                   {summary['mean_score']:.3f}")
        print(f"  Min:                    {summary['min_score']:.3f}")
        print(f"  Max:                    {summary['max_score']:.3f}")

    if summary.get('mean_absolute_error') is not None:
        print(f"\nAccuracy:")
        print(f"  Mean Absolute Error:    {summary['mean_absolute_error']:.3f}")

    print(f"\nSeverity Distribution:")
    for severity, count in summary['severity_distribution'].items():
        print(f"  {severity.capitalize():<20} {count}")

    print("="*60)


def evaluate_test_cases(
    category: str = None,
    verbose: bool = True,
    save: bool = True,
    output_path: str = None
) -> Dict:
    """
    Run batch evaluation on test cases.

    Args:
        category: Filter test cases by category (None for all)
        verbose: Print detailed progress
        save: Save results to file
        output_path: Custom output file path

    Returns:
        Dict with results and summary
    """
    print("\n" + "="*60)
    print("LLM-as-a-Judge Hallucination Evaluation")
    print("="*60)

    # Load test cases
    test_cases = get_test_cases(category)
    print(f"\nLoaded {len(test_cases)} test cases")
    if category:
        print(f"Filtered by category: {category}")

    # Initialize judge
    print("\nInitializing HallucinationJudge...")
    judge = HallucinationJudge()

    # Run evaluation
    print("\nStarting evaluation...\n")
    results = judge.batch_evaluate(test_cases, verbose=verbose)

    # Generate summary
    summary = judge.summarize_results(results)

    # Print results
    if verbose:
        print_results_table(results)
        print_summary(summary)

    # Save results
    if save:
        output_file = save_results(results, summary, output_path)
        return {"results": results, "summary": summary, "output_file": output_file}

    return {"results": results, "summary": summary}


def evaluate_single_response(
    query: str,
    response: str,
    context: str = None,
    verbose: bool = True
) -> Dict:
    """
    Evaluate a single RAG response.

    Args:
        query: User query
        response: RAG response (may include <Context> tags)
        context: Retrieved context (optional if inline context present)
        verbose: Print detailed results

    Returns:
        Evaluation result dict
    """
    judge = HallucinationJudge()
    result = judge.evaluate(query, response, context)

    if verbose:
        print("\n" + "="*60)
        print("Hallucination Evaluation Result")
        print("="*60)
        print(f"Query: {query}")
        print(f"\nHallucination Score: {result['hallucination_score']:.3f}")
        print(f"Severity: {result['severity']}")
        print(f"\nExplanation:")
        print(result['explanation'])

        if result.get('supported_claims'):
            print(f"\nSupported Claims:")
            for claim in result['supported_claims']:
                print(f"  ✓ {claim}")

        if result.get('unsupported_claims'):
            print(f"\nUnsupported Claims:")
            for claim in result['unsupported_claims']:
                print(f"  ✗ {claim}")

        print("="*60)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG system responses for hallucinations using LLM-as-a-Judge"
    )

    parser.add_argument(
        '--mode',
        choices=['batch', 'single'],
        default='batch',
        help='Evaluation mode: batch (test cases) or single (one response)'
    )

    parser.add_argument(
        '--category',
        choices=get_categories() + ['all'],
        default='all',
        help='Test case category filter (batch mode only)'
    )

    parser.add_argument(
        '--query',
        type=str,
        help='User query (single mode only)'
    )

    parser.add_argument(
        '--response',
        type=str,
        help='RAG response (single mode only)'
    )

    parser.add_argument(
        '--context',
        type=str,
        help='Retrieved context (single mode only)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output file path for results'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to file'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output'
    )

    parser.add_argument(
        '--list-categories',
        action='store_true',
        help='List available test case categories'
    )

    args = parser.parse_args()

    # Handle list categories
    if args.list_categories:
        print_test_case_summary()
        return 0

    # Batch evaluation mode
    if args.mode == 'batch':
        category = None if args.category == 'all' else args.category
        result = evaluate_test_cases(
            category=category,
            verbose=not args.quiet,
            save=not args.no_save,
            output_path=args.output
        )

        # Exit code based on results
        if result['summary']['failed_evaluations'] > 0:
            return 1
        return 0

    # Single evaluation mode
    elif args.mode == 'single':
        if not args.query or not args.response:
            print("Error: --query and --response are required in single mode", file=sys.stderr)
            return 1

        result = evaluate_single_response(
            query=args.query,
            response=args.response,
            context=args.context,
            verbose=not args.quiet
        )

        # Save if requested
        if not args.no_save:
            save_results([result], {}, args.output)

        return 0


if __name__ == "__main__":
    sys.exit(main())
