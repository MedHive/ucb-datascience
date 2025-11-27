from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# -----------------------------
# PROMPT TEMPLATE
# -----------------------------
prompt = """
Human: You are an expert assistant helping a human detect hallucinations in medical device claims.
Your task is to read the context (a 510(k) or medical-device document) and evaluate the claim.

Return a "hallucination score" between 0 and 1:
- Return 0   → if the claim is fully supported by the context.
- Return 1   → if the claim is NOT supported or contradicts the context.
- Return a value between 0 and 1 if support is unclear or partially aligned.
The higher the score, the more confident you are that the claim is NOT based on the context.

VERY IMPORTANT RULES:
- Use ONLY information from the context.  
- Do NOT use external medical knowledge.  
- Output ONLY the numeric score (float).  
- Do NOT add explanations or extra words.

<example>
Context: The Capnostream 20 is intended for CO2 and SPO2 monitoring and is for use with neonatal, pediatric and adult patients.
Statement: "The Capnostream 20 monitors CO2 and SpO2."
Assistant: 0.05
</example>

<example>
Context: The Capnostream 10 is intended for CO2 indications only.
Statement: "The Capnostream 10 monitors oxygen saturation."
Assistant: 1
</example>

<example>
Context: The FDA has determined that the Capnostream 20 is substantially equivalent and allowed marketing.
Statement: "The FDA rejected the Capnostream 20."
Assistant: 1
</example>

Context: {context}
Statement: {statement}

Assistant:
"""

# Create LangChain prompt template
prompt_template = PromptTemplate(
    template=prompt,
    input_variables=["context", "statement"],
)

def get_hallucination_score(context: str, statement: str, prompt_template: PromptTemplate, llm) -> float:
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