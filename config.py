# config.py
import os

# === MIMIC-IV 根目錄 ===
MIMIC4_ROOT = os.getenv("MIMIC4_ROOT", "/home/lab-206/MIMIC/mimic-iv-3.1/hosp/")

# === 編碼映射檔 ===
CONDITION_MAPPING_FILE = os.getenv("CONDITION_MAPPING_FILE", "./resources/CCSCM.csv")
PROCEDURE_MAPPING_FILE = os.getenv("PROCEDURE_MAPPING_FILE", "./resources/CCSPROC.csv")
DRUG_MAPPING_FILE = os.getenv("DRUG_MAPPING_FILE", "./resources/ATC.csv")

# === 快取路徑生成 ===
DATA_BASE_PATH = os.getenv("DATA_BASE_PATH", "./exp_data/ccscm_ccsproc")
def sample_dataset_path(dataset: str, task: str) -> str:
    return os.path.join(DATA_BASE_PATH, f"sample_dataset_{dataset}_{task}.pkl")

