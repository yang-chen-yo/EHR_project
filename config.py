# === config.py ===
import os

# === MIMIC-IV 根目錄 ===
MIMIC4_ROOT = os.getenv(
    "MIMIC4_ROOT", "/home/lab-206/MIMIC/mimic-iv-3.1/hosp/"
)

# === 編碼映射檔 ===
CONDITION_MAPPING_FILE = os.getenv(
    "CONDITION_MAPPING_FILE", "./resources/CCSCM.csv"
)
PROCEDURE_MAPPING_FILE = os.getenv(
    "PROCEDURE_MAPPING_FILE", "./resources/CCSPROC.csv"
)
DRUG_MAPPING_FILE = os.getenv(
    "DRUG_MAPPING_FILE", "./resources/ATC.csv"
)

# === 本地 UMLS 資料夾 ===
UMLS_DATA_DIR = os.getenv(
    "UMLS_DATA_DIR", "./resources/umls_data"
)

# === 資料快取路徑 ===
DATA_BASE_PATH = os.getenv(
    "DATA_BASE_PATH", "./exp_data/ccscm_ccsproc"
)


def sample_dataset_path(dataset: str, task: str) -> str:
    return os.path.join(
        DATA_BASE_PATH, f"sample_dataset_{dataset}_{task}.pkl"
    )
    
# Fusion & Scoring 參數
K_UMLS = int(os.getenv("K_UMLS", 5))
K_PUBMED = int(os.getenv("K_PUBMED", 5))
DECAY_LAMBDA = float(os.getenv("DECAY_LAMBDA", 0.1))
ALPHA_SIM = float(os.getenv("ALPHA_SIM", 0.7))
BETA_RECENCY = float(os.getenv("BETA_RECENCY", 0.3))

# === Gemini 嵌入模型設定 ===
GOOGLE_API_KEY = "AIzaSyDID8ktEZOsviITdt6QBUUMbbOb69i46Zk"
EMBED_MODEL_NAME = "text-embedding-004"  # 或  gemini-embedding-exp-03-07 也可
EMBED_BATCH_SIZE = 32
EMBED_TASK_TYPE = "RETRIEVAL_QUERY"  # 查詢用途

# RAG 模型設定
RAG_MODEL_NAME = os.getenv("RAG_MODEL_NAME", "huggingFaceH4/zephyr-7b-beta")
RAG_MAX_TOKENS = int(os.getenv("RAG_MAX_TOKENS", 2048))
