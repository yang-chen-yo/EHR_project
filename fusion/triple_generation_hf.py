# === fusion/triple_generation_hf.py ===
import json
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
    """
    # 1) 載入 tokenizer & 模型
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,    # 顯存允許就 4bit，否則改 8bit
        device_map="auto",
        trust_remote_code=True,
    )

    # 2) 建 pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )

    # 3) 構造 Prompt：強制 JSON block 並加上 few-shot 範例
    abstracts_str = "\n---\n".join(abstracts)
    umls_fact_str = "\n".join(umls_facts or [])
    prompt = f"""
SYSTEM: You are a medical knowledge-graph extraction assistant.  
Given a single patient context, UMLS related facts, and PubMed abstract snippets,  
output ONLY a JSON array of triples, wrapped inside ```json``` fences—nothing else.

USER: Patient Context and Timeline:
{patient_context}

UMLS Related Facts:
{umls_fact_str}
---
PubMed Abstract Snippets:
{abstracts_str}

REQUIRED ENTITY TYPES: Patient, Disease, Drug, Symptom, LabResult, Treatment, SideEffect, Severity  
REQUIRED RELATION TYPES:
  - HAS_DISEASE (patient→disease)  
  - USED_DRUG (patient→drug)  
  - TREATS (drug→disease)  
  - CAUSES_SIDE_EFFECT (drug→sideEffect)  
  - HAS_SYMPTOM (disease→symptom)  
  - HAS_LAB_RESULT (patient→labResult)  
  - RECEIVED_TREATMENT (patient→treatment)  
  - BEFORE / AFTER (time ordering)  

Example of desired output format:
```json
[
  {
    "head":"Patient:P123456",
    "head_type":"Patient",
    "relation":"HAS_DISEASE",
    "tail":"Disease:I10",
    "tail_type":"Disease",
    "timestamp":"2025-06-01",
    "source":"EHR"
  },
  {
    "head":"Patient:P123456",
    "head_type":"Patient",
    "relation":"USED_DRUG",
    "tail":"Drug:C09AA02",
    "tail_type":"Drug",
    "timestamp":"2025-07-10",
    "source":"EHR"
  }
]"""

    # 4) 生成並解析
    raw = generator(prompt)[0]["generated_text"]

    # 先統一還原雙大括號，之後再做正則比對
    cleaned = raw.replace("{{", "{").replace("}}", "}")

    import re, textwrap
    # 找最後一段 [ { "head": ... } ]   （允許中間有任意換行、空白）
    matches = re.findall(r"\[\s*{\s*\"head\"[\s\S]*?\]", cleaned)
    if not matches:
        print("[RAW OUTPUT]", raw[:300], "...")
        raise ValueError("Model did not return JSON array of triples")

    json_str = textwrap.dedent(matches[-1])  # 取最後一段結果
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print("[RAW OUTPUT]", raw[:300], "...")
        raise ValueError(f"無法解析 JSON: {e}")


# ——— Quick CLI test ——————————————————————————
if __name__ == "__main__":
    ctx = (
        "PatientID: P123456; Visit: 2025-06-01; Diagnoses: I10, E11.9; "
        "Medications: 2025-06-02 Lisinopril (C09AA02); "
        "Labs: BP=150/95, Glu=180 mg/dL; "
        "SideEffect: Dizziness 2025-06-05; Severity: ICU"
    )
    abs_list = [
        "In a randomized trial, lisinopril lowered systolic blood pressure by 15 mmHg...",
        "Metformin was generally well tolerated; rare episodes of dizziness observed...",
    ]
    umls_demo = [
        "lisinopril may_treat essential hypertension",
        "metformin may_cause dizziness",
    ]
    triples = generate_triples_local_llama2(ctx, abs_list, umls_demo)
    print("\n=== Triples ===")
    for t in triples:
        print(t)

