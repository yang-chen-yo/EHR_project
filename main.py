# main.py
from data.loader import load_mimic4_dataset, load_mappings, enrich_sample_with_names
from data.preprocess import preprocess_samples
from config import sample_dataset_path
from pyhealth.datasets.splitter import split_by_visit
from fusion.merger import fuse_and_score, merge_to_triples

import os
import pickle
from tqdm import tqdm

# === 控制開關 ===
DO_LOAD = True          # 載入原始資料
DO_PREPROCESS = True    # 前處理 raw data
DO_MAP = True           # mapping code -> name
DO_RAG = True           # 執行 RAG 檢索與合併


def main():
    # === 任務與資料集名稱 ===
    dataset_name = "mimic4"
    task_name = "drugrec"

    # === 資料儲存路徑 ===
    output_path = sample_dataset_path(dataset_name, task_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1) 載入與切分
    dataset = None
    if DO_LOAD:
        print(f"[INFO] Loading {dataset_name} for {task_name}...")
        dataset = load_mimic4_dataset(DO_LOAD, dataset_name, task_name)
        train_set, val_set, test_set = split_by_visit(dataset, ratios=(0.7,0.1,0.2), seed=42)
        print(f"[INFO] Split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    # 2) 前處理與 mapping
    processed = None
    if DO_PREPROCESS:
        if dataset is None:
            raise RuntimeError("Dataset not loaded")
        print("[INFO] Preprocessing...")
        raw_samples = preprocess_samples(dataset)
        if DO_MAP:
            print("[INFO] Loading mappings...")
            cond_map, proc_map, drug_map = load_mappings()
            processed = []
            for s in tqdm(raw_samples, desc="Mapping codes to names"):
                processed.append(enrich_sample_with_names(s, cond_map, proc_map, drug_map))
            print(f"[INFO] Enriched {len(processed)} samples.")
        else:
            processed = raw_samples
        # save preprocessed
        with open(output_path, 'wb') as f:
            pickle.dump(processed, f)
        print(f"[INFO] Saved preprocessed samples -> {output_path}")

    # 3) RAG 檢索與合併
    if DO_RAG:
        if processed is None:
            raise RuntimeError("No samples to RAG")
        # 取第一筆 sample 示範
        sample = processed[0]
        pid = sample['patient_id']
        # 拼 patient_context: 組合基本文字描述
        patient_context = f"PatientID: {pid}; " \
                          f"Conditions: {[c['name'] for group in sample['conditions'] for c in group]}; " \
                          f"Drugs: {[d['name'] for group in sample['drugs'] for d in group]}; " \
                          f"Procedures: {[p['name'] for group in sample['procedures'] for p in group]}"
        print(f"[DEBUG] Patient context: {patient_context}\n")
        # RAG 檢索
        fused = fuse_and_score(
            patient_text=patient_context,
            umls_dir=None,       # 若 UMLSClient 內部有 default 資料夾，可傳 None
            pubmed_email="test@example.com",
            patient_fields={
                'conditions': sample['conditions'],
                'procedures': sample['procedures'],
                'drugs': sample['drugs']
            }
        )
        # 列印檢索結果
        print("[DEBUG] UMLS hits:")
        for hit in fused['umls']:
            print(f"  CUI: {hit['cui']}, name: {hit['name']}, score: {hit['score']}")
        print("[DEBUG] PubMed hits:")
        for art in fused['pubmed']:
            print(f"  PMID: {art['pmid']}, score: {art['score']}\n    abstract: {art['abstract'][:100]}...")
        # 合併與生成 triples
        triples = merge_to_triples(
            patient_id=pid,
            fused=fused,
            patient_context=patient_context,
            patient_fields={
                'conditions': sample['conditions'],
                'procedures': sample['procedures'],
                'drugs': sample['drugs']
            },
            accumulate=True
        )
        print(f"[DEBUG] Generated {len(triples)} triples:")
        for t in triples:
            print(f"  {t}")

if __name__ == '__main__':
    main()
