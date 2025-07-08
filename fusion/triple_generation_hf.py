#fusion/triple_generation_hf
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
    Generate knowledge-graph triples using an int8-quantised Llama-2 chat model.

    Args:
      patient_context: English free-text of patient timeline.
      abstracts: List of PubMed abstract snippets.
      umls_facts: Optional UMLS fact strings.
      model_name: Model identifier from config.
      max_new_tokens: Generation length limit from config.

    Returns:
      List of dict triples with keys: head, head_type, relation, tail, tail_type, timestamp, source.
    """
    # 1) Load tokenizer and quantised model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto",
    )

    # 2) Create text-generation pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )

    # 3) Construct prompt
    abstracts_str = "\n---\n".join(abstracts)
    umls_fact_str = "\n".join(umls_facts or [])

    prompt = f"""
SYSTEM: You are a medical knowledge-graph extraction assistant. Given a single patient context, UMLS related facts, and PubMed abstract snippets, output ONLY a JSON array of triples.

USER: Patient Context and Timeline:
{patient_context}

UMLS Related Facts:
{umls_fact_str}
---
PubMed Abstract Snippets:
{abstracts_str}

REQUIRED ENTITY TYPES: [Patient, Disease, Drug, Symptom, LabResult, Treatment, SideEffect, Severity]
REQUIRED RELATION TYPES: [HAS_DISEASE, USED_DRUG, TREATS, CAUSES_SIDE_EFFECT, HAS_SYMPTOM, HAS_LAB_RESULT, RECEIVED_TREATMENT, HAS_SEVERITY, BEFORE, AFTER]
OUTPUT FORMAT: JSON array with keys: head, head_type, relation, tail, tail_type, timestamp (optional), source.
Example:
[
  {"head":"Patient:P123456", "head_type":"Patient", "relation":"HAS_DISEASE", "tail":"Disease:I10", "tail_type":"Disease", "timestamp":"2025-06-01", "source":"EHR"}
]
"""

    # 4) Generate and parse
    raw = generator(prompt)[0]["generated_text"]
    start = raw.find("[")
    if start == -1:
        raise ValueError("Model did not return JSON array of triples")
    return json.loads(raw[start:])


# Quick test
if __name__ == "__main__":
    ctx = (
        "PatientID: P123456; Visit: 2025-06-01; Diagnoses: I10, E11.9; "
        "Medications: 2025-06-02 Lisinopril (C09AA02); Labs: BP=150/95, Glu=180 mg/dL; "
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
    for t in triples:
        print(t)
