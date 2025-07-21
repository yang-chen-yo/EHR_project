# data/loader.py

import os
import csv
import pickle
from typing import List
from pyhealth.datasets import MIMIC4Dataset
from typing import List, Dict

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

def enrich_sample_with_names(
    sample: dict,
    cond_map: Dict[str, str],
    proc_map: Dict[str, str],
    drug_map: Dict[str, str]
) -> dict:
    def _convert(nested_codes, mapping: Dict[str, str]) -> List[List[dict]]:
        # 1) 若已是二層 list 且內層為 dict → 直接回傳
        if nested_codes and isinstance(nested_codes[0], list) and isinstance(nested_codes[0][0], dict):
            return nested_codes

        # 2) 若三層 list (多包一層) → 去掉外層
        if nested_codes and isinstance(nested_codes[0], list) and isinstance(nested_codes[0][0], list):
            nested_codes = nested_codes[0]
            # 去掉後再次檢查是否已 enrich
            if nested_codes and isinstance(nested_codes[0], list) and isinstance(nested_codes[0][0], dict):
                return nested_codes

        # 3) 若是一層 list of str → 包成兩層
        if nested_codes and isinstance(nested_codes[0], str):
            nested_codes = [nested_codes]

        # 4) 正常 mapping
        enriched: List[List[dict]] = []
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

def load_mimic4_dataset(
    load_processed: bool,
    dataset: str,
    task: str,
    dev: bool = True,
):
    """
    load_processed: 是否讀取快取
    dev=True: sample 模式 (~2%)；dev=False: 載入 full dataset
    """
    from config import sample_dataset_path, full_dataset_path

    cache_path = (
        sample_dataset_path(dataset, task)
        if dev
        else full_dataset_path(dataset, task)
    )

    # 如果要从自己存的 cache 读，就在这里返回
    if load_processed and os.path.exists(cache_path):
        print(f"Loading cached dataset for {task} from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # 强制不读旧的 PyHealth cache，always rebuild
    ds = MIMIC4Dataset(
        root=MIMIC4_ROOT,
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        code_mapping={
            "NDC": ("ATC", {"target_kwargs": {"level": 3}}),
            "ICD9CM": "CCSCM", "ICD9PROC": "CCSPROC",
            "ICD10CM": "CCSCM", "ICD10PROC": "CCSPROC",
        },
        dev=dev,
        refresh_cache=True,     # ← 加这一行
    )

    # 选对应任务的 fn
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

    # 如果是 full 模式，再存到自己的 cache
    if not dev:
        with open(cache_path, "wb") as f:
            pickle.dump(sample_dataset, f)
        print(f"Saved full dataset for {task} to {cache_path}")

    return sample_dataset
