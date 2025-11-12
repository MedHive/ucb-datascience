"""
Configuration for LLM-as-a-Judge Hallucination Evaluator
"""

import os
from typing import Dict, Any

# LLM Configuration
JUDGE_MODEL = "gpt-4o-mini"  # Same as GraphRAG
JUDGE_TEMPERATURE = 0.0  # Deterministic evaluation
JUDGE_MAX_TOKENS = 500

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Evaluation Thresholds
HALLUCINATION_THRESHOLDS = {
    "no_hallucination": 0.0,
    "minor_hallucination": 0.3,
    "moderate_hallucination": 0.6,
    "severe_hallucination": 0.9
}

# Output Configuration
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DEFAULT_OUTPUT_FORMAT = "json"  # json, csv, or both

# Evaluation Prompt Template
JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator tasked with detecting hallucinations in AI-generated responses.

**Task:** Evaluate whether the generated response contains hallucinations by comparing it against the retrieved context.

**Definitions:**
- **Hallucination:** Information in the response that is NOT supported by or contradicts the retrieved context
- **Grounded:** Information that is directly supported by the retrieved context
- **Score:** 0.0 (fully grounded, no hallucinations) to 1.0 (completely hallucinated)

**Input:**

User Query:
{query}

Retrieved Context:
{context}

Generated Response:
{response}

**Instructions:**
1. Extract claims from the generated response
2. For each claim, check if it's supported by the retrieved context
3. Identify any information that appears in the response but is NOT in the context
4. Calculate a hallucination score from 0.0 to 1.0:
   - 0.0: All claims are fully supported by context
   - 0.1-0.3: Minor unsupported details or reasonable inferences
   - 0.4-0.6: Significant claims without context support
   - 0.7-0.9: Majority of response is unsupported
   - 1.0: Response is entirely fabricated or contradicts context

**Output Format (JSON):**
{{
  "hallucination_score": <float between 0.0 and 1.0>,
  "explanation": "<detailed explanation of the score>",
  "unsupported_claims": ["<list of claims not supported by context>"],
  "supported_claims": ["<list of claims supported by context>"]
}}

Respond ONLY with valid JSON. Be precise and thorough in your evaluation.
"""

def get_judge_config() -> Dict[str, Any]:
    """Get judge model configuration"""
    return {
        "model": JUDGE_MODEL,
        "temperature": JUDGE_TEMPERATURE,
        "max_tokens": JUDGE_MAX_TOKENS,
        "api_key": OPENAI_API_KEY
    }
