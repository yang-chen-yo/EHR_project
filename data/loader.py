# data/loader.py

import os
import csv
import pickle
from pyhealth.datasets import MIMIC4Dataset
from data.task_fn import (
    drug_recommendation_mimic4_fn,
    mortality_prediction_mimic4_fn,
    readmission_prediction_mimic4_fn,
    length_of_stay_prediction_mimic4_fn
)
from config import (
    MIMIC4_ROOT,
    CONDITION_MAPPING_FILE,
    PROCEDURE_MAPPING_FILE,
    DRUG_MAPPING_FILE,
    sample_dataset_path
)

def load_mappings():
    """
    Load CSV mapping files into dictionaries.
    Returns:
        condition_dict, procedure_dict, drug_dict
    """
    condition_dict = {}
    with open(CONDITION_MAPPING_FILE, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            condition_dict[row['code']] = row['name'].lower()

    procedure_dict = {}
    with open(PROCEDURE_MAPPING_FILE, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            procedure_dict[row['code']] = row['name'].lower()

    drug_dict = {}
    with open(DRUG_MAPPING_FILE, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 只取 ATC 第 3 級群組
            if row.get('level') == '3.0':
                drug_dict[row['code']] = row['name'].lower()

    return condition_dict, procedure_dict, drug_dict

def enrich_sample_with_names(sample: dict,
                             cond_map: dict,
                             proc_map: dict,
                             drug_map: dict) -> dict:
    """
    將 sample 內的 conditions / procedures / drugs
    由 [['151','6',...], [...]] 轉成
    [[{'code':'151','name':'Essential hypertension'}, ...], [...]]
    """
    def _convert(nested_codes: List[List[str]], mapping: dict):
        enriched = []
        for code_list in nested_codes:
            enriched.append([
                {"code": c, "name": mapping.get(c, c).title()}
                for c in code_list
            ])
        return enriched

    sample["conditions"] = _convert(sample["conditions"], cond_map)
    sample["procedures"] = _convert(sample["procedures"], proc_map)
    sample["drugs"]      = _convert(sample["drugs"],      drug_map)
    return sample


def load_mimic4_dataset(load_processed: bool, dataset: str, task: str):
    """
    Load or build PyHealth MIMIC-IV sample dataset, and save per-task cache.
    Args:
        load_processed: if True and cached exists, load from pickle
        dataset: e.g. "mimic4"
        task: 'drugrec', 'mortality', 'readmission', or 'lenofstay'
    Returns:
        sample_dataset
    """
    cache_path = sample_dataset_path(dataset, task)

    if load_processed and os.path.exists(cache_path):
        print(f"Loading processed dataset for {task} from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    # 建構原始 MIMIC-IV Dataset
    ds = MIMIC4Dataset(
        root=MIMIC4_ROOT,
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        code_mapping={
            "NDC": ("ATC", {"target_kwargs": {"level": 3}}),
            "ICD9CM": "CCSCM",
            "ICD9PROC": "CCSPROC",
            "ICD10CM": "CCSCM",
            "ICD10PROC": "CCSPROC"
        },
        dev=True
    )

    # 選擇對應任務函式
    if task == "drugrec":
        fn = drug_recommendation_mimic4_fn
    elif task == "mortality":
        fn = mortality_prediction_mimic4_fn
    elif task == "readmission":
        fn = readmission_prediction_mimic4_fn
    elif task == "lenofstay":
        fn = length_of_stay_prediction_mimic4_fn
    else:
        raise ValueError(f"Unknown task: {task}")

    sample_dataset = ds.set_task(fn)

    # 快取至 pickle，避免重複前處理
    with open(cache_path, 'wb') as f:
        pickle.dump(sample_dataset, f)
    print(f"Saved processed dataset for {task} to {cache_path}")

    return sample_dataset
