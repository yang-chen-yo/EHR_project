# kg/triple.py
import pickle
from dataclasses import dataclass
from typing import Optional, Any, List, Dict
import config


# ===== 知識圖三元組結構 ======================================
@dataclass
class Triple:
    head: str
    head_type: str
    relation: str
    tail: str
    tail_type: str
    visit_id: str | None = None
    visit_date: str | None = None                       # ← 新增：就診日期 (YYYY-MM-DD)
    timestamp: Optional[str] = None       # 事件日期（可與 visit_date 不同）
    source: str = "EHR"
    weight: Optional[float] = None        # 用於存放邊權重


# ===== 將前處理樣本轉成三元組 =================================
def samples_to_triples(samples: List[Dict[str, Any]]) -> List[Triple]:
    """
    Convert pre-processed EHR samples into Triple list.

    必要鍵：
      - patient_id      str
      - visit_date      str (YYYY-MM-DD)
      - conditions      List[str]
      - procedures      List[str]
      - drugs           List[str]
    """
    triples: List[Triple] = []
    for rec in samples:
        pid        = rec["patient_id"]
        visit_date = rec.get("visit_date", "")        # 若缺則空字串
        # Patient → Disease
        for cond in rec.get("conditions", []):
            triples.append(Triple(
                head=f"Patient:{pid}", head_type="Patient",
                relation="HAS_DISEASE",
                tail=f"Disease:{cond}", tail_type="Disease",
                visit_date=visit_date, source="EHR"
            ))
        # Patient → Drug
        for drug in rec.get("drugs", []):
            triples.append(Triple(
                head=f"Patient:{pid}", head_type="Patient",
                relation="USED_DRUG",
                tail=f"Drug:{drug}", tail_type="Drug",
                visit_date=visit_date, source="EHR"
            ))
        # Patient → Treatment
        for proc in rec.get("procedures", []):
            triples.append(Triple(
                head=f"Patient:{pid}", head_type="Patient",
                relation="RECEIVED_TREATMENT",
                tail=f"Treatment:{proc}", tail_type="Treatment",
                visit_date=visit_date, source="EHR"
            ))
    return triples


# ===== 載入樣本 & 轉三元組 ====================================
def load_preprocessed_samples(dataset: str, task: str) -> List[Dict[str, Any]]:
    path = config.sample_dataset_path(dataset, task)
    with open(path, "rb") as f:
        return pickle.load(f)


def load_triples_from_samples(dataset: str, task: str) -> List[Triple]:
    samples = load_preprocessed_samples(dataset, task)
    return samples_to_triples(samples)

