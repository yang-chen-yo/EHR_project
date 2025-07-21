#!/usr/bin/env python3
# coding: utf-8
"""
data/patient_dataset.py
-----------------------
將所有三元組依 patient_id 分群 → 每位病人一張子圖 (torch_geometric.data.Data)。

濾除：
  • 該病人缺少目標任務標籤 (y = -1)
  • 子圖無任何節點 (極少見)

回傳：
    dataset : List[Data]  ─ 每筆含
        • x             : [num_nodes] 全域實體 ID (拿去做嵌入)
        • edge_index    : [2,E]
        • edge_type     : [E]
        • (edge_weight) : [E,1]（有開 use_time 時）
        • y             : [1] 或 [k] label
        • patient_id    : str (metadata)
    ent2id  : Dict[str,int]
    rel2id  : Dict[str,int]
    out_dim : int  (binary=1 / multilabel=k / multiclass= n_class)
"""

from pathlib import Path
import itertools, json, re
import torch
from torch_geometric.data import Data
from data.triple_loader import norm_patient
from data.label_loader import load_labels


# ────────────────────────────────────────────────
def build_patient_dataset(json_dir: str,
                          pkl_path: str,
                          task: str,
                          use_time: bool = True):
    # ------------ 讀取所有 JSON → 依病人分群 ------------ #
    groups = {}
    for fp in Path(json_dir).rglob("triples_*.json"):
        triples = json.load(open(fp))
        for t in triples:
            # 取病人 ID
            if t["head_type"].lower() == "patient":
                pid_raw = t["head"]
            elif t["tail_type"].lower() == "patient":
                pid_raw = t["tail"]
            else:
                continue
            pid = norm_patient(pid_raw)
            groups.setdefault(pid, []).append(t)

    # ------------ 建立全域實體 / 關係映射 ------------ #
    all_tr = list(itertools.chain.from_iterable(groups.values()))
    ents = {t["head"] for t in all_tr} | {t["tail"] for t in all_tr}
    rels = {t["relation"] for t in all_tr}
    ent2id = {e: i for i, e in enumerate(sorted(ents))}
    rel2id = {r: i for i, r in enumerate(sorted(rels))}

    # ------------ 讀取全域標籤 → y_all ------------ #
    y_all, _ = load_labels(pkl_path, ent2id, task)
    out_dim = y_all.shape[1] if y_all.dim() == 2 else 1

    dataset = []
    skip_no_label = skip_neg1 = 0

    # ------------ 為每位病人建子圖 ------------ #
    for pid, triples in groups.items():
        if pid not in ent2id:           # 不應發生
            continue

        y_vec = y_all[ent2id[pid]]
        if (y_vec < 0).any():           # -1 → skip
            skip_neg1 += 1; continue

        # 收集此病人涉及的全域節點
        node_set = {ent2id[t["head"]] for t in triples} | \
                   {ent2id[t["tail"]] for t in triples}
        if len(node_set) == 0:
            skip_no_label += 1; continue

        node_list = sorted(node_set)
        gid2lid = {gid: i for i, gid in enumerate(node_list)}

        # 邊
        src, dst, etype, ew = [], [], [], []
        for t in triples:
            h = ent2id[t["head"]]; d = ent2id[t["tail"]]
            src.append(gid2lid[h]); dst.append(gid2lid[d])
            etype.append(rel2id[t["relation"]])
            if use_time:
                ew.append([t.get("weight", 1.0)])

        data = Data(
            x=torch.tensor(node_list, dtype=torch.long),
            edge_index=torch.tensor([src, dst], dtype=torch.long),
            edge_type=torch.tensor(etype, dtype=torch.long),
            y=y_vec.unsqueeze(0) if y_vec.dim()==0 else y_vec.unsqueeze(0),
            patient_id=pid
        )
        if use_time:
            data.edge_weight = torch.tensor(ew, dtype=torch.float)

        dataset.append(data)

    print(f"◎ 病人子圖總數 = {len(dataset)} (跳過 -1 標籤 {skip_neg1}, 空圖 {skip_no_label})")
    return dataset, ent2id, rel2id, out_dim

