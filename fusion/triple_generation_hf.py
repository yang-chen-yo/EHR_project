import os
import sys
# ensure project root is on sys.path to locate config.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import re
import textwrap
from typing import List, Dict, Optional

from config import RAG_MODEL_NAME, RAG_MAX_TOKENS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def generate_triples_local_llama2(
    patient_context: str,
    abstracts: List[str],
    umls_facts: Optional[List[str]] = None,
    model_name: str = RAG_MODEL_NAME,
    max_new_tokens: int = RAG_MAX_TOKENS,
) -> List[Dict]:
    """
    Generate knowledge-graph triples with a quantised chat model.
    Post-process to strip prefixes from head and tail.
    """
    # 1) Load tokenizer & quantised model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
    )

    # 2) Build text-generation pipeline (only new text)
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        return_full_text=False,
        # Add stop sequence to end at closing fence
        
    )

    # 3) Construct Prompt
    abstracts_str = "\n---\n".join(abstracts)
    umls_fact_str = "\n".join(umls_facts or [])
    prompt = f"""
SYSTEM:
You are a Medical KG Extraction Assistant.
Use both the patient’s EHR context and the provided PubMed abstracts
to infer **all** relevant entity–relation triples.  

OUTPUT RULES:
- **Only** return a ```json``` code fence containing a JSON array of objects.
- **Do NOT** output any other text outside the code fence.
- The array **may contain multiple** objects—one per inferred triple.
- Each object must include: head, head_type, relation, tail, tail_type, timestamp (optional), source.

USER INPUT (do NOT modify):
Patient Context:
{patient_context}

PubMed Abstracts:
{abstracts_str}

ENTITY TYPES (nodes):
- Patient          # 病患ID
- Disease          # 疾病
- Drug             # 藥物
- Symptom          # 臨床症狀
- LabResult        # 實驗室檢驗結果（含單位）
- Treatment        # 治療方案（如手術／化療／放療）
- SideEffect       # 藥物副作用／不良反應
- Severity         # 病情嚴重度（如 ICU / 住院 / 門診）

RELATION TYPES (edges):
- HAS_DISEASE            (patient → disease)
- USED_DRUG              (patient → drug)
- TREATS                 (drug → disease)
- CAUSES_SIDE_EFFECT     (drug → sideEffect)
- HAS_SYMPTOM            (disease → symptom)
- HAS_LAB_RESULT         (patient → labResult)
- RECEIVED_TREATMENT     (patient → treatment)
- BEFORE / AFTER         (time ordering)

INFERENCE REQUIREMENTS:
1. Use **both** EHR and abstracts to infer triples—do **not** copy values from the Example.
2. There **may be multiple** triples for one patient; include **all** you can infer.
3. Each triple must set `"source":"EHR"` or `"source":"PubMed"`.
4. If a value (e.g. lab number) appears in both EHR and PubMed, choose the most specific one.

FORMAT EXAMPLE (placeholders only—do NOT copy these values):
```json
[
  {
    "head": "Patient:<PatientID>",
    "head_type": "Patient",
    "relation": "<RELATION>",
    "tail": "<EntityType>:<Code_or_Name_or_Value>",
    "tail_type": "<EntityType>",
    "timestamp": "<YYYY-MM-DD>",
    "source": "<EHR_or_PubMed>"
  },
  {
    "head": "Patient:<PatientID>",
    "head_type": "Patient",
    "relation": "<RELATION>",
    "tail": "<EntityType>:<Code_or_Name_or_Value>",
    "tail_type": "<EntityType>",
    "timestamp": "<YYYY-MM-DD>",
    "source": "<EHR_or_PubMed>"
  }
]"""

    # 4) 生成並解析
    raw = generator(prompt)[0]["generated_text"]

    # 5) Extract JSON array by finding first '[' and last ']'
    start = raw.find('[')
    end = raw.rfind(']')
    if start == -1 or end == -1 or end <= start:
        print("[RAW OUTPUT]", raw)
        raise ValueError("Model did not return a complete JSON array of triples")
    json_str = raw[start:end+1]

    # 6) Parse JSON
    try:
        triples = json.loads(json_str)
    except json.JSONDecodeError as e:
        print("[RAW OUTPUT]", raw)
        raise ValueError(f"JSON parsing error: {e}")

    # 7) Post-process: strip prefix before ':' in head and tail
    cleaned = []
    for t in triples:
        h = t.get('head', '')
        ta = t.get('tail', '')
        t['head'] = h.split(':', 1)[1] if ':' in h else h
        t['tail'] = ta.split(':', 1)[1] if ':' in ta else ta
        cleaned.append(t)
    return cleaned


# Quick CLI Test
if __name__ == "__main__":
    ctx = (
        "PatientID: P123456; Visit: 2025-06-01; Diagnoses: I10, E11.9; "
        "Medications: 2025-06-02 Lisinopril (C09AA02); "
        "Labs: BP=160/95, Glu=180 mg/dL; SideEffect: Dizziness 2025-06-05; Severity: ICU"
    )
    abs_list = ["Lisinopril lowered BP by 15 mmHg in hypertension patients."]
    triples = generate_triples_local_llama2(ctx, abs_list, None)
    print("=== Triples ===")
    for t in triples:
        print(t)

