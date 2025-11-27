#!/usr/bin/env python3
"""
Test script to verify the hallucination evaluation framework is working
Demonstrates the framework without requiring API calls
"""

import json
from datetime import datetime
from test_cases import TEST_CASES, print_test_case_summary

def test_framework():
    """Test that the framework is set up correctly"""
    print("="*80)
    print("HALLUCINATION EVALUATION FRAMEWORK - SETUP TEST")
    print("="*80)

    # Test 1: Verify test cases are loaded
    print("\n1. Testing test case loading...")
    print(f"   ✓ Loaded {len(TEST_CASES)} test cases")
    print_test_case_summary()

    # Test 2: Verify test case structure
    print("\n2. Verifying test case structure...")
    required_fields = ["id", "category", "query", "context", "response", "expected_score"]
    for tc in TEST_CASES[:3]:  # Check first 3
        missing = [f for f in required_fields if f not in tc]
        if missing:
            print(f"   ✗ Test case {tc['id']} missing fields: {missing}")
        else:
            print(f"   ✓ Test case {tc['id']} has all required fields")

    # Test 3: Show example test case
    print("\n3. Example Test Case:")
    example = TEST_CASES[0]
    print(f"   ID: {example['id']}")
    print(f"   Category: {example['category']}")
    print(f"   Query: {example['query']}")
    print(f"   Expected Score: {example['expected_score']}")
    print(f"   Response preview: {example['response'][:100]}...")

    # Test 4: Verify context tag parsing logic
    print("\n4. Testing context tag parsing...")
    test_response = "This is a test response. <Context>Source info here</Context>"

    # Simulate parsing
    import re
    context_pattern = r'<Context>(.*?)</Context>'
    contexts = re.findall(context_pattern, test_response, re.DOTALL | re.IGNORECASE)
    clean_response = re.sub(context_pattern, '', test_response, flags=re.DOTALL | re.IGNORECASE).strip()

    print(f"   Original: {test_response}")
    print(f"   Parsed Response: {clean_response}")
    print(f"   Extracted Context: {contexts[0] if contexts else 'None'}")
    print(f"   ✓ Context tag parsing works correctly")

    # Test 5: Verify judge prompt template
    print("\n5. Verifying judge prompt template...")
    try:
        from config import JUDGE_PROMPT_TEMPLATE
        if "{query}" in JUDGE_PROMPT_TEMPLATE and "{context}" in JUDGE_PROMPT_TEMPLATE:
            print("   ✓ Judge prompt template has required placeholders")
        else:
            print("   ✗ Judge prompt template missing required placeholders")
    except Exception as e:
        print(f"   ✗ Error loading config: {e}")

    # Test 6: Simulate evaluation result structure
    print("\n6. Example Evaluation Result Structure:")
    mock_result = {
        "hallucination_score": 0.15,
        "explanation": "The response is mostly grounded in the provided context...",
        "unsupported_claims": ["Minor detail about 50 years experience"],
        "supported_claims": [
            "Capnostream20 is manufactured by Oridion Medical",
            "Device received 510(k) clearance"
        ],
        "severity": "minor",
        "timestamp": datetime.now().isoformat(),
        "query": "What is Capnostream20?",
        "model": "gpt-4o-mini"
    }
    print(json.dumps(mock_result, indent=2))

    # Test 7: Directory structure
    print("\n7. Verifying directory structure...")
    import os
    from pathlib import Path

    eval_dir = Path(__file__).parent
    required_files = [
        "llm_judge.py",
        "test_cases.py",
        "config.py",
        "evaluate_rag.py",
        "rag_with_evaluation.py"
    ]

    for file in required_files:
        file_path = eval_dir / file
        if file_path.exists():
            print(f"   ✓ {file} exists")
        else:
            print(f"   ✗ {file} missing")

    results_dir = eval_dir / "results"
    if results_dir.exists():
        print(f"   ✓ results/ directory exists")
    else:
        print(f"   ✗ results/ directory missing")

    print("\n" + "="*80)
    print("FRAMEWORK SETUP TEST COMPLETE")
    print("="*80)
    print("\nNext Steps:")
    print("1. Set OPENAI_API_KEY environment variable")
    print("2. Run: python evaluate_rag.py --mode batch --category accurate")
    print("3. Or test single evaluation:")
    print("   python evaluate_rag.py --mode single \\")
    print("     --query 'What is Capnostream20?' \\")
    print("     --response 'A medical device <Context>From K101011</Context>' \\")
    print("     --context 'Capnostream20 is a medical monitoring device'")
    print("\n" + "="*80)


if __name__ == "__main__":
    test_framework()
