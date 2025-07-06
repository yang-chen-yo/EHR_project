from data.loader import load_mimic4_dataset
from data.preprocess import preprocess_samples
from config import sample_dataset_path
from pyhealth.datasets.splitter import split_by_visit

import os
import pickle
from tqdm import tqdm

# === 控制開關 ===
DO_LOAD = True
DO_PREPROCESS = True

def main():
    # === 任務與資料集名稱 ===
    dataset_name = "mimic4"
    task_name = "drugrec"  # 可選："mortality", "readmission", "lenofstay", "drugrec"

    # === 資料儲存路徑 ===
    output_path = sample_dataset_path(dataset_name, task_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dataset = None
    if DO_LOAD:
        # === 資料載入 ===
        print(f"[INFO] Loading {dataset_name} dataset for task: {task_name}...")
        dataset = load_mimic4_dataset(DO_LOAD, dataset_name, task_name)
        print("[INFO] Dataset loaded.")

        # === 切分資料集 ===
        train_set, val_set, test_set = split_by_visit(
            dataset,
            ratios=(0.7, 0.1, 0.2),
            seed=42
        )
        print("[INFO] Train/Val/Test size:")
        print("Train:", len(train_set))
        print("Val:  ", len(val_set))
        print("Test: ", len(test_set))

    if DO_PREPROCESS:
        if dataset is None:
            raise ValueError("[ERROR] Dataset not loaded. Please set DO_LOAD = True")

        # === 資料前處理 ===
        print("[INFO] Preprocessing samples...")
        raw_samples = preprocess_samples(dataset)
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

