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

