# fusion/triple_generation_hf.py
"""Generate triples via local LLM (逐篇摘要)，並清楚列出 prompt 規則。"""

import json
from typing import List, Dict, Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from config import RAG_MODEL_NAME, RAG_MAX_TOKENS

# ──────────────────────────────────────────────────────────
# 1) 量化與模型初始化（只執行一次）
# ──────────────────────────────────────────────────────────
torch.cuda.empty_cache()
quant_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(RAG_MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    RAG_MODEL_NAME,
    quantization_config=quant_cfg,
    device_map="auto",
    trust_remote_code=True,
)

# 停止條件：遇到 EOS 就結束生成
class StopOnEOS(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return input_ids[0, -1].item() == tokenizer.eos_token_id
stop_list = StoppingCriteriaList([StopOnEOS()])


def generate_triples_local_llama2(
    patient_context: str,
    abstracts: List[str],
    umls_facts: Optional[List[str]] = None,
    max_new_tokens: int = RAG_MAX_TOKENS,
) -> List[Dict]:
    """
    逐篇摘要呼叫 LLM，輸出純 JSON array，並回傳 Triple-like dict list。

    輸入：
      - patient_context: 病患基本文字描述
      - abstracts: PubMed 摘要清單

    輸出：約略格式：
      [{ 'head': str, 'head_type': str, 'relation': str,
         'tail': str, 'tail_type': str,
         'timestamp': str (optional), 'source': 'PubMed' }]
    """
    all_triples: List[Dict] = []
    max_input_len = model.config.max_position_embeddings - max_new_tokens
    MAX_ABS_LEN = 200  # 每段摘要取前200字

    for abstract in abstracts:
        # 先截短摘要
        abs_short = abstract[:MAX_ABS_LEN]
        prompt = f"""
SYSTEM:
You are a Medical KG Extraction Assistant.

PATIENT CONTEXT:
{patient_context}

PUBMED ABSTRACT (shortened):
{abs_short}

ENTITY TYPES:
- Patient: 病患 ID
- Disease: 疾病
- Drug: 藥物
- Symptom: 臨床症狀
- LabResult: 實驗室檢驗結果（含單位）
- Treatment: 治療方案（如手術／化療／放療）
- SideEffect: 藥物副作用／不良反應
- Severity: 病情嚴重度（如 ICU / 住院 / 門診）

RELATION TYPES:
- HAS_DISEASE (patient→disease)
- USED_DRUG (patient→drug)
- TREATS (drug→disease)
- CAUSES_SIDE_EFFECT (drug→sideEffect)
- HAS_SYMPTOM (disease→symptom)
- HAS_LAB_RESULT (patient→labResult)
- RECEIVED_TREATMENT (patient→treatment)
- BEFORE / AFTER (time ordering)

INFERENCE RULES:
1. Use both the EHR context and this single abstract. Do NOT invent facts.
2. Return at most 2 most important triples for this abstract.
3. Output ONLY a JSON array (no other text).
4. Follow the EXACT output format below.

OUTPUT FORMAT EXAMPLE:
```json
[
  {
    "head": "Patient:10000032",
    "head_type": "Patient",
    "relation": "HAS_DISEASE",
    "tail": "Disease:Hypertension",
    "tail_type": "Disease",
    "timestamp": "2025-01-01",  # optional
    "source": "PubMed"
  }
]
```

BEGIN JSON
```json
["""

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            max_length=max_input_len,
        ).to(model.device)

        with torch.no_grad():
            out_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                stopping_criteria=stop_list,
            )
        raw = tokenizer.decode(out_ids[0], skip_special_tokens=True)

        # 擷取 JSON block：從 '[' 到最後 ']' 
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end <= start:
            print("[ERROR] No JSON block found:\n", raw[:200])
            continue
        json_str = raw[start:end+1]
        try:
            triples = json.loads(json_str)
        except json.JSONDecodeError:
            print("[ERROR] JSON parse failed:\n", json_str[:200])
            continue

        # 清理 prefix 並限量
        for t in triples[:2]:
            if ":" in t.get("head", ""):
                t["head"] = t["head"].split(":",1)[1]
            if ":" in t.get("tail", ""):
                t["tail"] = t["tail"].split(":",1)[1]
            all_triples.append(t)

    return all_triples

