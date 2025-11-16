# Lightweight test for get_hallucination_score using a mock LLM and existing prompt
from prompt import prompt_template


# Simple mock LLM that returns a text response when called with a prompt string
class MockLLM:
    def __init__(self, response_text: str):
        self.response_text = response_text

    def __call__(self, prompt_text: str, **kwargs):
        # mimic returning a plain string as an LLM would
        return self.response_text


def parse_score_from_text(text: str) -> float:
    raw_output = text.strip()
    try:
        return float(raw_output)
    except Exception:
        print(f"Could not parse LLM response: {raw_output}")
        return 1.0


if __name__ == '__main__':
    context = (
        "The Capnostream 20 is intended for CO2 and SPO2 monitoring and is for use with neonatal, pediatric and adult patients."
    )
    statement = "The Capnostream 20 monitors CO2 and SpO2."

    # Render the prompt using the PromptTemplate (from langchain 0.x)
    rendered = prompt_template.format(context=context, statement=statement)

    # Case A: clean numeric LLM response
    llm = MockLLM("0.05")
    resp = llm(rendered)
    print("Rendered prompt (truncated):", rendered[:120].replace('\n', ' '), "...")
    print("Mock numeric response -> score:", parse_score_from_text(resp))

    # Case B: noisy response
    llm2 = MockLLM("I think the answer is 0.1")
    resp2 = llm2(rendered)
    print("Mock noisy response -> score:", parse_score_from_text(resp2))

    # Case C: invalid response
    llm3 = MockLLM("not a number")
    resp3 = llm3(rendered)
    print("Mock invalid response -> score:", parse_score_from_text(resp3))
