from langchain import PromptTemplate, LLMChain
from typing import Any


def get_hallucination_score(context: str, statement: str, prompt_template: Any, llm) -> float:
    """
    Sends the medical-device context + statement to the LLM
    and returns a hallucination score between 0 and 1.
    """
    llm_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=False)

    # Ask the model
    response = llm_chain({
        "context": context,
        "statement": statement
    })

    raw_output = response["text"].strip()

    # Parse the float
    try:
        score = float(raw_output)
    except Exception:
        print(f"Could not parse LLM response: {raw_output}")
        score = 1.0   # default to hallucination if parsing fails

    return score