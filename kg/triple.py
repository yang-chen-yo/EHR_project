#kg/triple.py
import pickle
from dataclasses import dataclass
from typing import Optional, Any, List, Dict
from config import sample_dataset_path

# === 知識圖三元組結構 ===
@dataclass
class Triple:
    head: str
    head_type: str
    relation: str
    tail: str
    tail_type: str
    timestamp: Optional[str] = None
    source: str = "EHR"


def samples_to_triples(samples: List[Dict[str, Any]]) -> List[Triple]:
    """
    Convert preprocessed EHR samples into initial EHR-based triples.

    Each sample dict must include:
      - patient_id: str
      - conditions: List[str]
      - procedures: List[str]
      - drugs: List[str]
      - visit_time: Optional[str]

    Returns a list of Triple with relations:
      - Patient->Disease, Patient->Drug, Patient->Treatment
    """
    triples: List[Triple] = []
    for rec in samples:
        pid = rec['patient_id']
        visit_time = rec.get('visit_time')
        # Patient - Disease
        for cond in rec.get('conditions', []):
            triples.append(Triple(
                head=f"Patient:{pid}", head_type="Patient",
                relation="HAS_DISEASE", tail=f"Disease:{cond}", tail_type="Disease",
                timestamp=visit_time, source="EHR"
            ))
        # Patient - Drug
        for drug in rec.get('drugs', []):
            triples.append(Triple(
                head=f"Patient:{pid}", head_type="Patient",
                relation="USED_DRUG", tail=f"Drug:{drug}", tail_type="Drug",
                timestamp=visit_time, source="EHR"
            ))
        # Patient - Treatment (procedures)
        for proc in rec.get('procedures', []):
            triples.append(Triple(
                head=f"Patient:{pid}", head_type="Patient",
                relation="RECEIVED_TREATMENT", tail=f"Treatment:{proc}", tail_type="Treatment",
                timestamp=visit_time, source="EHR"
            ))
    return triples


def load_preprocessed_samples(dataset: str, task: str) -> List[Dict[str, Any]]:
    """
    Load preprocessed samples from pickle, using config.sample_dataset_path.

    Args:
        dataset: dataset name (e.g. "mimic4").
        task: task name (e.g. "mortality").
    Returns:
        List of sample dicts.
    """
    path = sample_dataset_path(dataset, task)
    with open(path, 'rb') as f:
        samples = pickle.load(f)
    return samples


def load_triples_from_samples(dataset: str, task: str) -> List[Triple]:
    """
    Convenience: Load preprocessed samples then convert to triples.
    """
    samples = load_preprocessed_samples(dataset, task)
    return samples_to_triples(samples)
