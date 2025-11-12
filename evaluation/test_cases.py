"""
Synthetic Test Cases for Hallucination Detection
Medical device queries with ground truth labels
"""

from typing import List, Dict

# Synthetic test cases for medical device queries
TEST_CASES = [
    # Case 1: Fully grounded response (NO hallucination)
    {
        "id": "test_001",
        "category": "accurate",
        "query": "What is Capnostream20?",
        "context": """
        Capnostream 20 bedside monitor is indicated for use in monitoring respiration
        via carbon dioxide (CO2) and oxygen saturation (SpO2), as well as vital signs
        including blood pressure and pulse rate. The device is manufactured by
        Oridion Medical 1987 Ltd and received 510(k) clearance under K101011.
        """,
        "response": """Capnostream20 is a bedside monitor used for monitoring respiration
        via carbon dioxide and oxygen saturation, as well as vital signs including blood
        pressure and pulse rate. <Context>Capnostream 20 bedside monitor is described in
        510(k) document K101011, manufactured by Oridion Medical 1987 Ltd.</Context>""",
        "expected_score": 0.0,
        "notes": "Response fully supported by context"
    },

    # Case 2: Complete hallucination
    {
        "id": "test_002",
        "category": "hallucinated",
        "query": "What is the Capnostream20 used for?",
        "context": """
        Capnostream 20 bedside monitor is indicated for use in monitoring respiration
        via carbon dioxide (CO2) and oxygen saturation (SpO2), as well as vital signs.
        """,
        "response": """Capnostream20 is primarily used for performing surgical procedures
        and can also function as a defibrillator in emergency situations. It was approved
        by the FDA in 2005. <Context>Mentioned in medical device documentation.</Context>""",
        "expected_score": 1.0,
        "notes": "Response contradicts context - device is a monitor, not surgical tool"
    },

    # Case 3: Partial hallucination (mixed accurate and fabricated info)
    {
        "id": "test_003",
        "category": "partial",
        "query": "Who manufactures the Capnostream20?",
        "context": """
        The Capnostream 20 is manufactured by Oridion Medical 1987 Ltd and received
        510(k) clearance under K101011.
        """,
        "response": """The Capnostream20 is manufactured by Oridion Medical 1987 Ltd,
        a leading medical device company with over 50 years of experience and offices
        in 25 countries worldwide. <Context>Manufactured by Oridion Medical 1987 Ltd,
        documented in K101011.</Context>""",
        "expected_score": 0.4,
        "notes": "Manufacturer is correct, but company details (50 years, 25 countries) are fabricated"
    },

    # Case 4: Accurate with reasonable inference
    {
        "id": "test_004",
        "category": "accurate",
        "query": "Is the Capnostream20 FDA cleared?",
        "context": """
        Capnostream 20 bedside monitor received 510(k) clearance under K101011.
        """,
        "response": """Yes, the Capnostream20 is FDA cleared. It received 510(k) clearance,
        which means the FDA determined it to be substantially equivalent to a predicate device.
        <Context>Received 510(k) clearance under K101011.</Context>""",
        "expected_score": 0.1,
        "notes": "Main claim supported; explanation of 510(k) is reasonable inference"
    },

    # Case 5: Wrong device entirely (severe hallucination)
    {
        "id": "test_005",
        "category": "hallucinated",
        "query": "What type of device is the Capnostream20?",
        "context": """
        Capnostream 20 bedside monitor is indicated for use in monitoring respiration
        via carbon dioxide (CO2) and oxygen saturation (SpO2).
        """,
        "response": """The Capnostream20 is an implantable cardiac pacemaker designed to
        regulate heart rhythm in patients with arrhythmias. <Context>Medical device
        documentation K101011.</Context>""",
        "expected_score": 1.0,
        "notes": "Completely wrong device type - monitor vs pacemaker"
    },

    # Case 6: Accurate response without context tags (testing parser robustness)
    {
        "id": "test_006",
        "category": "accurate",
        "query": "What parameters does the Capnostream20 monitor?",
        "context": """
        Capnostream 20 bedside monitor is indicated for use in monitoring respiration
        via carbon dioxide (CO2) and oxygen saturation (SpO2), as well as vital signs
        including blood pressure and pulse rate.
        """,
        "response": """The Capnostream20 monitors respiration via carbon dioxide (CO2),
        oxygen saturation (SpO2), blood pressure, and pulse rate.""",
        "expected_score": 0.0,
        "notes": "No context tags but fully accurate",
        "parse_inline_context": False
    },

    # Case 7: Minor hallucination with correct core info
    {
        "id": "test_007",
        "category": "partial",
        "query": "What measurements can Capnostream20 take?",
        "context": """
        Capnostream 20 bedside monitor monitors CO2, SpO2, blood pressure, and pulse rate.
        """,
        "response": """The Capnostream20 can measure carbon dioxide (CO2), oxygen saturation
        (SpO2), blood pressure, pulse rate, and body temperature. <Context>Monitors CO2,
        SpO2, blood pressure, and pulse rate as described in K101011.</Context>""",
        "expected_score": 0.2,
        "notes": "Body temperature is not mentioned in context"
    },

    # Case 8: Ambiguous/vague response (moderate hallucination)
    {
        "id": "test_008",
        "category": "partial",
        "query": "What is the Capnostream20 used for?",
        "context": """
        Capnostream 20 bedside monitor is indicated for use in monitoring respiration
        via carbon dioxide (CO2) and oxygen saturation (SpO2).
        """,
        "response": """The Capnostream20 is a state-of-the-art medical device used in
        hospitals and clinics for comprehensive patient care and monitoring of all vital
        signs throughout surgery and recovery. <Context>Used for monitoring mentioned
        in 510(k) documentation.</Context>""",
        "expected_score": 0.5,
        "notes": "Overgeneralizes - context only mentions specific parameters, not 'all vital signs'"
    },

    # Case 9: Accurate with multiple specific claims
    {
        "id": "test_009",
        "category": "accurate",
        "query": "Tell me about the Capnostream20 device clearance",
        "context": """
        The Capnostream 20 bedside monitor, manufactured by Oridion Medical 1987 Ltd,
        received 510(k) clearance under K101011. The device is indicated for monitoring
        respiration via CO2 and SpO2, as well as vital signs including blood pressure
        and pulse rate.
        """,
        "response": """The Capnostream20 received 510(k) clearance under submission number
        K101011. It was manufactured by Oridion Medical 1987 Ltd. The device is cleared
        for monitoring respiration (CO2 and SpO2) and vital signs (blood pressure and pulse).
        <Context>510(k) K101011, Oridion Medical 1987 Ltd, monitors CO2, SpO2, blood
        pressure, pulse rate.</Context>""",
        "expected_score": 0.0,
        "notes": "Multiple specific claims, all supported"
    },

    # Case 10: No information in context (should indicate uncertainty)
    {
        "id": "test_010",
        "category": "hallucinated",
        "query": "What is the price of the Capnostream20?",
        "context": """
        Capnostream 20 bedside monitor is indicated for use in monitoring respiration
        via carbon dioxide (CO2) and oxygen saturation (SpO2).
        """,
        "response": """The Capnostream20 typically costs between $15,000 and $20,000 depending
        on configuration and includes a 2-year warranty. <Context>Device specifications
        from K101011.</Context>""",
        "expected_score": 0.9,
        "notes": "Price and warranty info not in context - should say 'information not available'"
    },

    # Case 11: Correct refusal to answer
    {
        "id": "test_011",
        "category": "accurate",
        "query": "What is the price of the Capnostream20?",
        "context": """
        Capnostream 20 bedside monitor is indicated for use in monitoring respiration
        via carbon dioxide (CO2) and oxygen saturation (SpO2).
        """,
        "response": """I don't have information about the price of the Capnostream20 in
        the available documentation. <Context>Device information from 510(k) K101011 does
        not include pricing.</Context>""",
        "expected_score": 0.0,
        "notes": "Correctly indicates lack of information"
    },

    # Case 12: Conflation of multiple devices (moderate hallucination)
    {
        "id": "test_012",
        "category": "partial",
        "query": "What does the Capnostream20 do?",
        "context": """
        Capnostream 20 bedside monitor monitors CO2 and SpO2. The document also mentions
        the BIS monitor which measures depth of anesthesia using EEG signals.
        """,
        "response": """The Capnostream20 is a monitor that measures CO2, SpO2, and depth
        of anesthesia using EEG signals. <Context>Capnostream 20 monitors CO2 and SpO2,
        BIS monitor uses EEG for anesthesia depth.</Context>""",
        "expected_score": 0.6,
        "notes": "Conflates two different devices - EEG is from BIS monitor, not Capnostream20"
    },
]


def get_test_cases(category: str = None) -> List[Dict]:
    """
    Get test cases, optionally filtered by category.

    Args:
        category: Filter by category (accurate, hallucinated, partial) or None for all

    Returns:
        List of test case dictionaries
    """
    if category:
        return [tc for tc in TEST_CASES if tc["category"] == category]
    return TEST_CASES


def get_test_case_by_id(test_id: str) -> Dict:
    """Get a specific test case by ID"""
    for tc in TEST_CASES:
        if tc["id"] == test_id:
            return tc
    raise ValueError(f"Test case {test_id} not found")


def get_categories() -> List[str]:
    """Get list of test case categories"""
    return list(set(tc["category"] for tc in TEST_CASES))


def print_test_case_summary():
    """Print summary of test cases"""
    categories = {}
    for tc in TEST_CASES:
        cat = tc["category"]
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\nTest Case Summary:")
    print(f"Total test cases: {len(TEST_CASES)}")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    print()


if __name__ == "__main__":
    print_test_case_summary()
    print("\nSample Test Case:")
    print(f"ID: {TEST_CASES[0]['id']}")
    print(f"Category: {TEST_CASES[0]['category']}")
    print(f"Query: {TEST_CASES[0]['query']}")
    print(f"Expected Score: {TEST_CASES[0]['expected_score']}")
