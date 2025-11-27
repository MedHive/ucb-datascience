import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_ID = "google/gemma-2-9b-it"  # base model with tokenizer files
ADAPTER_ID = "neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1"  # <-- set your adapter repo here

USE_CPU = True

INSTRUCTION = (
    "Generate Cypher statement to query a graph database. "
    "Use only the provided relationship types and properties in the schema. \n"
    "Schema: {schema} \n Question: {question}  \n Cypher output: "
)

def prepare_chat_prompt(question: str, schema: str):
    return [{
        "role": "user",
        "content": INSTRUCTION.format(schema=schema, question=question),
    }]

def _postprocess_output_cypher(output_cypher: str) -> str:
    # Strip any explanation / code fences / language tags
    partition_by = "**Explanation:**"
    output_cypher, _, _ = output_cypher.partition(partition_by)
    output_cypher = output_cypher.strip("`\n")
    output_cypher = output_cypher.lstrip("cypher\n")
    output_cypher = output_cypher.strip("`\n ")
    return output_cypher

def read_multiline_schema() -> str:
    print("Paste your SCHEMA, then Ctrl-D (macOS/Linux) or Ctrl-Z (Windows) to finish:")
    buf = []
    try:
        while True:
            line = input()
            buf.append(line)
    except EOFError:
        pass
    schema = "\n".join(buf).strip()
    if not schema:
        # minimal default to avoid empty schema
        schema = "(:Submission {k_number, decision_date, decision})-[:SUBJECT_DEVICE]->(:Device {device_name})"
        print("\n[Info] Empty schema. Using default:\n", schema, file=sys.stderr)
    return schema

if USE_CPU:
    device = "cpu"
    torch_dtype = torch.float32
else:
    if torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float16
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    else:
        device = "cpu"
        torch_dtype = torch.float32

print(f"# Loading base model: {BASE_ID} ...", file=sys.stderr)
tokenizer = AutoTokenizer.from_pretrained(BASE_ID, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    torch_dtype=torch_dtype,
    attn_implementation="eager",
    low_cpu_mem_usage=True,
)
print(f"# Applying PEFT adapter: {ADAPTER_ID} ...", file=sys.stderr)
model = PeftModel.from_pretrained(base_model, ADAPTER_ID)

model.to(device)
model.eval()

schema = read_multiline_schema()
print("Enter your QUESTION: ", end="", flush=True)
try:
    question = input().strip()
except EOFError:
    question = ""

if not question:
    question = "What is the subject device name for K101995?"
    print("\n[Info] Empty question. Using default:", question, file=sys.stderr)

chat = prepare_chat_prompt(question=question, schema=schema)
prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

MAX_INPUT_LEN = 512
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=MAX_INPUT_LEN,
)
inputs = {k: v.to(device) for k, v in inputs.items()}

gen_kwargs = {
    "max_new_tokens": 96,              # small output
    "do_sample": False,                # greedy
    "use_cache": True,                # BIG memory saver on MPS
    "pad_token_id": tokenizer.eos_token_id,
}

print("# Generating...", file=sys.stderr)
with torch.no_grad():
    tokens = model.generate(**inputs, **gen_kwargs)
    tokens = tokens[:, inputs["input_ids"].shape[1]:]
    raw = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    outputs = [_postprocess_output_cypher(o) for o in raw]

print("\n# Cypher:")
for o in outputs:
    print(o)


# (:Submission {k_number, decision_date, decision})-[:SUBJECT_DEVICE]->(:Device {device_name})
# What is the subject device name for K101995?