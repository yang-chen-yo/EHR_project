import json
import re
import time
from typing import List, Dict
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError

# ★★ 事先配置你的 API Key ★★
genai.configure(api_key="AIzaSyCew69BjF4VwjlbqTZMbOAK27jjvU24EBI")
MODEL_ID = "gemma-3-4b-it"


def _sanitize_raw(text: str) -> str:
    # 清理常見 JSON 格式錯誤
    text = text.strip()
    # 移除 code fence
    text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.IGNORECASE)
    # 分號轉逗號（key/value 之間用逗號）
    text = re.sub(r';\s*(?=")', ',', text)
    # 修正多重冒號
    text = re.sub(r'::', ':', text)
    # 取代 head/tail 嵌錯
    text = re.sub(r'"(head|tail)"\s*:\s*"(\w+)"\s*:\s*"', r'"\1": "\2:', text)
    # 移除多餘屬性或標籤
    text = re.sub(r'"([^"\n]+:[^"\n]+)"\s*:\s*"<[^">]+>"', r'"\1"', text)
    text = re.sub(r'"([^"\n]+:[^"\n]+)"\s*:\s*"[^"]+"', r'"\1"', text)
    # 修正 source 標記
    text = re.sub(r'"source"\s*:\s*"(Patient Context|PubMed Abstracts)"', '"source": "RAG"', text)
    # 移除物件/陣列尾逗號
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return text


def _extract_json(raw: str) -> List[Dict]:
    raw = _sanitize_raw(raw)
    # 嘗試直接解析
    try:
        data = json.loads(raw)
        data_list = data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        # 分段提取
        data_list = []
        for block in re.findall(r"\[[^\]]+\]", raw, flags=re.S):
            cleaned = _sanitize_raw(block)
            try:
                part = json.loads(cleaned)
                if isinstance(part, list):
                    data_list.extend(part)
            except json.JSONDecodeError:
                continue
    # 驗證並修正 timestamp
    valid_list: List[Dict] = []
    for item in data_list:
        ts = item.get('timestamp')
        if ts:
            try:
                year = int(ts.split('-')[0])
                if year < 1900 or year > datetime.now().year:
                    item['timestamp'] = None
            except Exception:
                item['timestamp'] = None
        valid_list.append(item)
    return valid_list
    

def generate_triples_via_gemma(
    patient_context: str,
    abstracts: List[str],
    patient_id: str = "",
    batch_size: int = 5,
    max_output_tokens: int = 4096,
) -> List[Dict]:
    """
    呼叫 Gemini API 產生三元組，並帶 retry 機制
    """
    sys_prompt = """You are a Medical Knowledge-Graph Extraction Assistant.
Use ONLY the information found in the patient EHR context and the PubMed abstracts; do NOT hallucinate or rely on outside knowledge.

For EACH abstract, extract EXACTLY 3 triples:
  1. One whose tail_type is Symptom or LaboratoryResult
  2. One whose tail_type is Treatment or Procedure
  3. One whose tail_type is AdverseEffect

Return a SINGLE JSON array of objects, adhering strictly to the following rules:
1. Output ONLY the raw JSON array—no markdown, no code fences (```), no explanatory text.
2. Use EXACT wording from the abstracts/EHR for entity strings.
3. Use exactly ONE colon ':' in each key-value pair—never '::' or multiple colons.
4. Ensure every object has ALL required fields in the exact order shown below.
5. Ensure the JSON is valid and properly formatted (e.g., correct brackets, commas, and quotes).

Required schema:
[
  {
    "head": "<EntityType>:<ExactTextFromAbstractOrEHR>",
    "head_type": "<EntityType>",
    "relation": "<RelationLabelAsSeen>",
    "tail": "<EntityType>:<ExactTextFromAbstractOrEHR>",
    "tail_type": "<EntityType>",
    "timestamp": "<YYYY-MM-DD>",
    "source": "<EHR_or_PubMed>"
  }
]

Entity types:
- Patient • Disease • Drug • Symptom • LaboratoryResult
- Treatment • Procedure • AdverseEffect • Severity

Relation types:
- Patient–Disease • Patient–Drug • Disease–Drug • Drug–AdverseEffect
- Disease–Symptom • Patient–LaboratoryResult • Patient–Treatment • Temporal

Example (for structure only, do NOT copy values):
[
  {
    "head": "Patient:123456",
    "head_type": "Patient",
    "relation": "Patient–Disease",
    "tail": "Disease:Diabetes Mellitus",
    "tail_type": "Disease",
    "timestamp": "2024-05-01",
    "source": "PubMed"
  }
]
"""

    model = genai.GenerativeModel(MODEL_ID)
    all_triples: List[Dict] = []

    for i in range(0, len(abstracts), batch_size):
        batch = abstracts[i : i + batch_size]
        abstracts_txt = "\n\n---\n\n".join(batch)
        user_prompt = f"""Patient Context:
PatientID: {patient_id}
{patient_context}

PubMed Abstracts:
{abstracts_txt}

Instructions:
1. For each abstract, extract exactly 3 triples (Symptom/Lab, Treatment/Procedure, AdverseEffect).
2. Use entities **only from the patient context or the abstracts**—do NOT hallucinate or invent values.
3. Use the PatientID only if the abstract clearly refers to the patient’s condition or treatment.
4. Output a clean JSON array, no explanations or markdown.
"""

        raw = ""
        # 三次重試機制
        for attempt in range(1):
            try:
                resp = model.generate_content(
                    contents=[
                        {"role": "model", "parts": [{"text": sys_prompt}]},
                        {"role": "user",  "parts": [{"text": user_prompt}]},
                    ],
                    generation_config={
                        "temperature": 0.0,
                        "max_output_tokens": max_output_tokens,
                    },
                )
                raw = resp.text
                break
            except GoogleAPIError as e:
                msg = str(e).lower()
                if e.code == 429 or "quota" in msg:
                    print(f"[WARN] Gemma batch {i//batch_size+1} quota hit, retry {attempt+1}/3 after 3s")
                    time.sleep(1)
                    continue
                print(f"[ERROR] Gemma batch {i//batch_size+1} error: {e}")
                break
            except Exception as e:
                print(f"[WARN] Gemma batch {i//batch_size+1} unexpected: {e}")
                break
        else:
            print(f"[ERROR] Gemma batch {i//batch_size+1} failed 3 times, skip")
            continue

        time.sleep(1)
        triples = _extract_json(raw)
        for t in triples:
            # 淨化欄位
            t["source"] = "RAG"
            t["head"] = t.get("head", patient_id).split(":",1)[-1]
            t["tail"] = t.get("tail", "").split(":",1)[-1]
        all_triples.extend(triples)

    return all_triples

