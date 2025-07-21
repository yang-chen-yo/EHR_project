#!/usr/bin/env python
# main_bigData.py — 支援 sample/full 兩種模式 + 批次 RAG + 進度條
import os
import pickle
import gc
import random
from tqdm import tqdm

from data.loader import load_mimic4_dataset, load_mappings, enrich_sample_with_names
from data.preprocess import preprocess_samples
from config import sample_dataset_path, full_dataset_path, UMLS_DATA_DIR
from fusion.merger import fuse_and_score_progress, merge_to_triples

# === 開關 & 參數 ===
DO_LOAD       = True     # 讀 MIMIC-IV 並存 raw.pkl
DO_PREPROCESS = True     # 前處理 + mapping 並存 preprocessed.pkl
DO_RAG        = True     # UMLS+PubMed+GPT → triples

DATASET_NAME    = "mimic4"
TASK_NAME       = "mortality"
USE_FULL        = False        # False→sample 模式；True→full 模式
DEV_PATIENT_NUM = 1000         # sample 模式下取的病人數
BATCH_SIZE      = 100          # RAG 每批處理筆數


def main():
    path_fn   = full_dataset_path if USE_FULL else sample_dataset_path
    base      = os.path.splitext(path_fn(DATASET_NAME, TASK_NAME))[0]
    raw_path  = base + "_raw.pkl"
    proc_path = base + "_preprocessed.pkl"

    # 1) 載入 / 建 raw dataset
    if DO_LOAD:
        print("[INFO] 開始載入原始資料集...", flush=True)
        dataset = load_mimic4_dataset(
            load_processed=False,
            dataset=DATASET_NAME,
            task=TASK_NAME,
            dev=not USE_FULL,
        )
        print(f"[INFO] 原始資料集載入完成，共 {len(dataset)} 筆患者")
        with open(raw_path, "wb") as f:
            pickle.dump(dataset, f)
        print(f"[INFO] Saved raw dataset → {raw_path}")
    else:
        with open(raw_path, "rb") as f:
            dataset = pickle.load(f)
        print(f"[INFO] Loaded raw dataset ← {raw_path}, 共 {len(dataset)} 筆患者")

    # 2) 前處理
    if DO_PREPROCESS:
        print("[INFO] 開始前處理並 enrich sample with names...", flush=True)
        processed = []
        for sample in tqdm(dataset, desc="Preprocessing", unit="patient"):
            enriched = enrich_sample_with_names(sample)
            processed.append(enriched)
        processed = preprocess_samples(processed)
        with open(proc_path, "wb") as f:
            pickle.dump(processed, f)
        print(f"[INFO] Saved preprocessed dataset → {proc_path}")
    else:
        with open(proc_path, "rb") as f:
            processed = pickle.load(f)
        print(f"[INFO] Loaded preprocessed dataset ← {proc_path}, 共 {len(processed)} 筆患者")

    # 3) sample 模式下 **手動抽樣**
    if not USE_FULL:
        random.seed(42)
        processed = random.sample(processed if DO_PREPROCESS else dataset, DEV_PATIENT_NUM)
        print(f"[INFO] After sampling: {len(processed)} patients")

    # 4) 分批 RAG
    if DO_RAG:
        all_triples = []
        total_batches = (len(processed) + BATCH_SIZE - 1) // BATCH_SIZE
        for idx in range(0, len(processed), BATCH_SIZE):
            batch = processed[idx : idx + BATCH_SIZE]
            batch_id = idx // BATCH_SIZE + 1
            print(f"\n=== RAG Batch {batch_id}/{total_batches} | size={len(batch)} ===")
            pbar = tqdm(batch, desc="RAG", unit="patient")
            for sample in pbar:
                pid, vid = sample["patient_id"], sample["visit_id"]
                visit_date = sample.get("visit_date", "")
                patient_context = (
                    f"PatientID: {pid}; VisitDate: {visit_date}; "
                    f"Conditions: {[c['name'] for grp in sample['conditions'] for c in grp]}; "
                    f"Drugs: {[d['name'] for grp in sample['drugs'] for d in grp]}; "
                    f"Procedures: {[p['name'] for grp in sample['procedures'] for p in grp]}"
                )
                fused = fuse_and_score_progress(
                    patient_id=pid,
                    visit_id=vid,
                    patient_text=patient_context,
                    umls_dir=UMLS_DATA_DIR,
                    pubmed_email="you@example.com",
                    patient_fields={
                        "conditions": sample["conditions"],
                        "procedures": sample["procedures"],
                        "drugs": sample["drugs"],
                    },
                    pbar=pbar,
                )
                triples = merge_to_triples(
                    patient_id=pid,
                    visit_id=vid,
                    fused=fused,
                    patient_context=patient_context,
                    patient_fields={
                        "conditions": sample["conditions"],
                        "procedures": sample["procedures"],
                        "drugs": sample["drugs"],
                    },
                    visit_date=visit_date,
                    accumulate=True,
                )
                all_triples.extend(triples)
            pbar.close()
        print(f"[INFO] 完成 RAG，總 triples = {len(all_triples)}")
    else:
        print("[INFO] 跳過 RAG 步驟")

if __name__ == "__main__":
    main()

