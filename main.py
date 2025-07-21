# main.py
import os, pickle
from tqdm import tqdm

from data.loader import load_mimic4_dataset, load_mappings, enrich_sample_with_names
from data.preprocess import preprocess_samples
from config import sample_dataset_path, UMLS_DATA_DIR
from pyhealth.datasets.splitter import split_by_visit
from fusion.merger import fuse_and_score_progress, merge_to_triples   # <- 已改函式簽名

# === 開關 ===
DO_LOAD       = True    # 讀 MIMIC-IV 並存 raw.pkl
DO_PREPROCESS = True    # 前處理 + mapping 並存 preprocessed.pkl
DO_RAG        = False    # UMLS+PubMed+GPT → triples

def main():
    dataset_name = "mimic4"
    task_name    = "drugrec"

    # ---------- 檔案路徑 ----------
    base = os.path.splitext(sample_dataset_path(dataset_name, task_name))[0]
    raw_path        = base + "_raw.pkl"
    processed_path  = base + "_preprocessed.pkl"

    # ---------- 1) 載入 / 建立資料 ----------
    if DO_LOAD:
        dataset = load_mimic4_dataset(True, dataset_name, task_name)
        with open(raw_path, "wb") as f: pickle.dump(dataset, f)
    else:
        with open(raw_path, "rb") as f: dataset = pickle.load(f)

    train_set, val_set, test_set = split_by_visit(dataset, ratios=(0.7, 0.1, 0.2), seed=42)

    # ---------- 2) 前處理 + Code → Name ----------
    if DO_PREPROCESS:
        raw_samples = preprocess_samples(dataset)
        cond_map, proc_map, drug_map = load_mappings()
        processed = [
            enrich_sample_with_names(s, cond_map, proc_map, drug_map)
            for s in tqdm(raw_samples, desc="Mapping codes")
        ]
        with open(processed_path, "wb") as f: pickle.dump(processed, f)
    else:
        with open(processed_path, "rb") as f: processed = pickle.load(f)

    # ---------- 3) UMLS / PubMed / GPT ----------
    if DO_RAG and processed:
        all_triples = []
        pbar = tqdm(processed, desc="RAG 處理", unit="patient")
        for sample in pbar:
            pid         = sample["patient_id"]
            visit_id   = sample["visit_id"]
            visit_date = sample.get("visit_date", "")           # ★ 讀取就診日期
            tqdm.write(f"[DEBUG] Start RAG for PatientID={pid} ({visit_date})")

            patient_context = (
                f"PatientID: {pid}; VisitDate: {visit_date}; "
                f"Conditions: {[c['name'] for grp in sample['conditions'] for c in grp]}; "
                f"Drugs: {[d['name'] for grp in sample['drugs'] for d in grp]}; "
                f"Procedures: {[p['name'] for grp in sample['procedures'] for p in grp]}"
            )

            fused = fuse_and_score_progress(
                patient_id=pid,
                visit_id=visit_id,
                patient_text = patient_context,
                umls_dir     = UMLS_DATA_DIR,
                pubmed_email = "boy7770730@gmail.com",
                patient_fields = {
                    "conditions": sample["conditions"],
                    "procedures": sample["procedures"],
                    "drugs":      sample["drugs"],
                },
                pbar = pbar
            )

            triples = merge_to_triples(
                patient_id      = pid,
                visit_id        = visit_id,
                fused           = fused,
                patient_context = patient_context,
                patient_fields  = {
                    "conditions": sample["conditions"],
                    "procedures": sample["procedures"],
                    "drugs":      sample["drugs"],
                },
                visit_date      = visit_date,             # ★ 必填參數
                accumulate      = True,
            )
            tqdm.write(f"[DEBUG] {pid} 產生 {len(triples)} triples")
            all_triples.extend(triples)

        pbar.close()
        print(f"[INFO] 完成 RAG，總 triples = {len(all_triples)}")

if __name__ == "__main__":
    main()

