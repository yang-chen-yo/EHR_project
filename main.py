#main.py
from data.loader import load_mimic4_dataset
from data.preprocess import preprocess_samples
from config import sample_dataset_path

import os
import pickle
from tqdm import tqdm

# === 控制開關 ===
DO_LOAD = True
DO_PREPROCESS = True

def main():
    # === 任務與資料集名稱 ===
    dataset_name = "mimic4"
    task_name = "mortality"  # 可選："mortality", "readmission", "lenofstay", "drug"

    # === 資料儲存路徑 ===
    output_path = sample_dataset_path(dataset_name, task_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dataset = None
    if DO_LOAD:
        # === 資料載入 ===
        print(f"[INFO] Loading {dataset_name} dataset for task: {task_name}...")
        dataset = load_mimic4_dataset(task=task_name)
        print("[INFO] Dataset loaded.")
        print("[INFO] Train/Val/Test size:")
        print("Train:", len(dataset.train))
        print("Val:", len(dataset.dev))
        print("Test:", len(dataset.test))

    if DO_PREPROCESS:
        if dataset is None:
            raise ValueError("[ERROR] Dataset not loaded. Please set DO_LOAD = True")

        # === 資料前處理 ===
        print("[INFO] Preprocessing samples...")
        raw_samples = preprocess_samples(dataset)
        # 使用 tqdm 顯示處理進度
        samples = []
        for sample in tqdm(raw_samples, desc="Processing samples"):
            samples.append(sample)
        print(f"[INFO] Processed {len(samples)} samples.")

        # === 儲存前處理結果 ===
        with open(output_path, "wb") as f:
            pickle.dump(samples, f)
        print(f"[INFO] Saved processed samples to {output_path}")

if __name__ == "__main__":
    main()
