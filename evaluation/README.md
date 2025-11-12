# Hallucination Evaluation Framework
## LLM-as-a-Judge Approach for RAG Systems

This framework implements an LLM-based hallucination detection system for evaluating DocumentRAG and GraphRAG outputs, following the approach described in the AWS blog on RAG hallucination detection.

## 📁 Project Structure

```
evaluation/
├── __init__.py                 # Package initialization
├── config.py                   # Configuration and prompt templates
├── llm_judge.py               # Core HallucinationJudge class
├── test_cases.py              # 12 synthetic test cases
├── evaluate_rag.py            # Standalone batch evaluation CLI
├── rag_with_evaluation.py     # Real-time evaluation wrapper
├── test_framework.py          # Framework verification script
├── README.md                  # This file
└── results/                   # Evaluation output directory
```

## 🚀 Features

### 1. LLM-as-a-Judge Evaluator (`llm_judge.py`)
- Uses GPT-4o-mini to detect hallucinations in RAG responses
- Assigns hallucination scores from 0.0 (fully grounded) to 1.0 (completely hallucinated)
- Parses inline `<Context>` XML tags from responses
- Provides detailed explanations with supported/unsupported claims
- Classifies severity: none, minor, moderate, severe

### 2. Synthetic Test Dataset (`test_cases.py`)
- 12 carefully crafted test cases for medical device queries
- Categories:
  - **Accurate (5 cases)**: Fully grounded responses
  - **Hallucinated (3 cases)**: Fabricated or contradictory information
  - **Partial (4 cases)**: Mix of accurate and unsupported claims
- Each test case includes:
  - Query, context, response with `<Context>` tags
  - Expected hallucination score (ground truth)
  - Detailed notes explaining the evaluation rationale

### 3. Batch Evaluation CLI (`evaluate_rag.py`)
- Evaluate multiple test cases in batch
- Filter by category
- Generate JSON reports with statistics
- Summary metrics: mean score, severity distribution, MAE

### 4. Real-time Evaluation Wrapper (`rag_with_evaluation.py`)
- Wrap any RAG system with evaluation capabilities
- Display hallucination scores after each query
- Track evaluation history across session

### 5. Modified RAG Systems
Both RAG systems now include inline context tags in responses:

**DocumentRAG** (`testRAG.py`):
```python
response = rag_query(query, include_context_tags=True)
# Output: "Answer text <Context>Retrieved from 3 chunks from K060065.pdf.txt</Context>"
```

**GraphRAG** (`testMDKGRAG.py`):
```python
result = graph_rag.query(question, include_context_tags=True)
# Output: "Answer <Context>Retrieved from 5 chunks (avg relevance: 0.85), entities: Diabetes, Metformin...</Context>"
```

## ⚙️ Setup

### Prerequisites
- Python 3.8+
- OpenAI API key
- Required packages: `openai`, `neo4j`, `langchain`, etc.

### Environment Configuration
```bash
export OPENAI_API_KEY="your-api-key-here"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_PASSWORD="your-password"
```

### Verify Installation
```bash
cd evaluation
python test_framework.py
```

This will verify:
- ✓ All 12 test cases loaded correctly
- ✓ Test case structure is valid
- ✓ Context tag parsing works
- ✓ Judge prompt template is configured
- ✓ All required files exist

## 📖 Usage

### 1. Batch Evaluation

Evaluate all test cases:
```bash
python evaluate_rag.py --mode batch
```

Evaluate specific category:
```bash
python evaluate_rag.py --mode batch --category accurate
python evaluate_rag.py --mode batch --category hallucinated
python evaluate_rag.py --mode batch --category partial
```

Save results to custom location:
```bash
python evaluate_rag.py --mode batch --output my_results.json
```

Quiet mode (minimal output):
```bash
python evaluate_rag.py --mode batch --quiet
```

### 2. Single Response Evaluation

Evaluate a single RAG response:
```bash
python evaluate_rag.py --mode single \
  --query "What is Capnostream20?" \
  --response "Capnostream20 is a medical device... <Context>From K101011.pdf</Context>" \
  --context "Capnostream 20 bedside monitor is indicated for..."
```

### 3. Real-time Evaluation in RAG Systems

Wrap your RAG system with evaluation:

```python
from evaluation.rag_with_evaluation import EvaluatedRAGWrapper
from DocumentRAG.testMDKGRAG import ImprovedGraphRAG

# Initialize your RAG system
graph_rag = ImprovedGraphRAG(driver, llm, embeddings)

# Wrap with evaluation
evaluated_rag = EvaluatedRAGWrapper(graph_rag, enable_evaluation=True)

# Query with automatic hallucination detection
result = evaluated_rag.query("What is metformin?")

# Access evaluation
print(f"Hallucination Score: {result['evaluation']['hallucination_score']}")
print(f"Severity: {result['evaluation']['severity']}")
```

### 4. List Available Test Cases

```bash
python evaluate_rag.py --list-categories
```

Output:
```
Test Case Summary:
Total test cases: 12
  accurate: 5
  hallucinated: 3
  partial: 4
```

## 📊 Evaluation Output

### Console Output (Batch Mode)
```
================================================================================
ID           Category     Score    Expected   Severity     Query
================================================================================
test_001     accurate     0.05     0.0        none         What is Capnostream20?...
test_002     hallucinated 0.92     1.0        severe       What is the Capnostream20...
test_003     partial      0.38     0.4        moderate     Who manufactures the Cap...
...
================================================================================

EVALUATION SUMMARY
================================================================================
Total cases:              12
Successful evaluations:   12
Failed evaluations:       0

Hallucination Scores:
  Mean:                   0.342
  Min:                    0.000
  Max:                    0.950

Accuracy:
  Mean Absolute Error:    0.067

Severity Distribution:
  None                    5
  Minor                   2
  Moderate                3
  Severe                  2
  Error                   0
================================================================================
```

### JSON Output
```json
{
  "metadata": {
    "timestamp": "2025-11-12T10:30:00",
    "total_cases": 12,
    "evaluator": "LLM-as-a-Judge (GPT-4o-mini)"
  },
  "summary": {
    "total_cases": 12,
    "successful_evaluations": 12,
    "mean_score": 0.342,
    "severity_distribution": {...}
  },
  "results": [
    {
      "test_case_id": "test_001",
      "query": "What is Capnostream20?",
      "hallucination_score": 0.05,
      "explanation": "The response is fully grounded...",
      "supported_claims": [...],
      "unsupported_claims": [],
      "severity": "none",
      "expected_score": 0.0,
      "score_error": 0.05
    },
    ...
  ]
}
```

## 🎯 Judge Evaluation Criteria

The LLM judge evaluates responses based on:

1. **Claim Extraction**: Identifies factual claims in the response
2. **Context Verification**: Checks if each claim is supported by retrieved context
3. **Hallucination Detection**: Flags unsupported or contradictory information
4. **Scoring**:
   - 0.0: All claims fully supported
   - 0.1-0.3: Minor unsupported details
   - 0.4-0.6: Significant unsupported claims
   - 0.7-0.9: Majority unsupported
   - 1.0: Entirely fabricated

## 🔬 Test Case Examples

### Example 1: Fully Grounded (Score: 0.0)
```python
{
  "query": "What is Capnostream20?",
  "context": "Capnostream 20 is a bedside monitor for monitoring CO2 and SpO2...",
  "response": "Capnostream20 is a bedside monitor for respiratory monitoring. <Context>From K101011</Context>",
  "expected_score": 0.0
}
```

### Example 2: Complete Hallucination (Score: 1.0)
```python
{
  "query": "What is the Capnostream20 used for?",
  "context": "Capnostream 20 is a bedside monitor...",
  "response": "Capnostream20 is a surgical device and defibrillator. <Context>K101011</Context>",
  "expected_score": 1.0  # Contradicts context - monitor vs surgical tool
}
```

### Example 3: Partial Hallucination (Score: 0.4)
```python
{
  "query": "Who manufactures Capnostream20?",
  "context": "Manufactured by Oridion Medical 1987 Ltd...",
  "response": "Made by Oridion Medical, a company with 50 years experience in 25 countries. <Context>K101011</Context>",
  "expected_score": 0.4  # Manufacturer correct, but company details fabricated
}
```

## 🧪 Testing Pipeline

1. **Verify Framework**: `python test_framework.py`
2. **Test Accurate Cases**: `python evaluate_rag.py --mode batch --category accurate`
3. **Test Hallucinated Cases**: `python evaluate_rag.py --mode batch --category hallucinated`
4. **Test Partial Cases**: `python evaluate_rag.py --mode batch --category partial`
5. **Full Evaluation**: `python evaluate_rag.py --mode batch`
6. **Review Results**: Check `results/evaluation_results_*.json`

## 📈 Integration with RAG Systems

### DocumentRAG Integration
```python
from DocumentRAG import testRAG

# Query returns response with context tags
response, chunks = testRAG.rag_query("What is...", include_context_tags=True)
print(response)  # "Answer <Context>Source info</Context>"
```

### GraphRAG Integration
```python
from DocumentRAG.testMDKGRAG import ImprovedGraphRAG

graph_rag = ImprovedGraphRAG(driver, llm, embeddings)
result = graph_rag.query("What is...", include_context_tags=True)
print(result['answer'])  # "Answer <Context>Graph source info</Context>"
```

## 🎓 Key Concepts

### Hallucination Types Detected
1. **Fabrication**: Entirely invented information
2. **Contradiction**: Information that contradicts context
3. **Overgeneralization**: Correct core info but excessive claims
4. **Conflation**: Mixing information from different sources incorrectly
5. **Extrapolation**: Going beyond what context supports

### Severity Thresholds
- **None** (0.0): Perfect groundedness
- **Minor** (0.1-0.3): Small unsupported details, reasonable inferences
- **Moderate** (0.4-0.6): Significant unsupported claims
- **Severe** (0.7-1.0): Majority or entirely hallucinated

## 🔧 Configuration

Edit `config.py` to customize:
- Judge model (default: `gpt-4o-mini`)
- Temperature (default: 0.0 for deterministic evaluation)
- Severity thresholds
- Prompt template
- Output formats

## 📝 Next Steps

1. **Run Initial Evaluation**:
   ```bash
   export OPENAI_API_KEY="your-key"
   python evaluate_rag.py --mode batch
   ```

2. **Analyze Results**: Review generated JSON reports in `results/`

3. **Iterate on RAG Systems**: Use evaluation insights to improve:
   - Chunking strategies
   - Retrieval accuracy
   - Context formatting
   - Prompt engineering

4. **Build Test Dataset**: Add domain-specific test cases to `test_cases.py`

5. **Integrate Real-time Evaluation**: Wrap production RAG systems with `EvaluatedRAGWrapper`

## 🐛 Troubleshooting

### "OPENAI_API_KEY not set" Error
```bash
export OPENAI_API_KEY="sk-your-key-here"
# Verify:
echo $OPENAI_API_KEY
```

### Import Errors
```bash
cd ucb-datascience/evaluation
python -c "import llm_judge; print('OK')"
```

### Neo4j Connection Issues (for GraphRAG)
```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_PASSWORD="your-password"
# Test connection:
python -c "from neo4j import GraphDatabase; print('OK')"
```

## 📚 References

- AWS Blog: [Detect Hallucinations for RAG-based Systems](https://aws.amazon.com/blogs/machine-learning/detect-hallucinations-for-rag-based-systems/)
- Approach: LLM-based hallucination detection (Approach 1)
- Implementation: GPT-4o-mini as judge with structured JSON output

## 🎯 Success Metrics

- ✅ 12 diverse test cases covering accurate, hallucinated, and partial responses
- ✅ Automated evaluation pipeline for batch processing
- ✅ Real-time evaluation wrapper for production use
- ✅ Context tag integration in both DocumentRAG and GraphRAG
- ✅ Comprehensive evaluation reports with severity classification
- ✅ Ground truth comparison for accuracy measurement

---

**Status**: Framework implementation complete and verified ✓

For questions or issues, refer to test outputs in `results/` directory.
