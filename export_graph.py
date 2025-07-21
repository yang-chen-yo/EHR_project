#!/usr/bin/env python3
# coding: utf-8
"""
依指定任務把所有三元組建成靜態大圖，存成 graphs/graph_<task>.pt
"""
import argparse, os, torch
from data.graph_builder import build_graph
import config                                       # ← 單檔匯入

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--task", required=True, choices=config.TASKS)
    pa.add_argument("--json_dir", default="triples_output")
    pa.add_argument("--out_dir",  default="graphs")
    a = pa.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    out_path = os.path.join(a.out_dir, f"graph_{a.task}.pt")
    if os.path.exists(out_path):
        print("✓ 已存在：", out_path)
        return

    conf = config.TASKS[a.task]
    data = build_graph(a.json_dir, conf["pkl"], a.task)
    # 檢查死亡率標籤 0/1 分佈
    zero_count = int((data.y == 0).sum().item())
    one_count  = int((data.y == 1).sum().item())
    print(f"死亡標籤分佈 ➔ 0: {zero_count} 筆, 1: {one_count} 筆")
    # 在 torch.save(...) 之前插入
    print("🛈  節點總數:", data.num_nodes)
    print("🛈  邊總數  :", data.edge_index.size(1))
    print("🛈  不同實體類型:", len({n.split(':')[0] for n in data.x_dict.keys()}) if hasattr(data,'x_dict') else '單類型')
    print("🛈  關係種類   :", data.num_relations)
    print("🛈  sample 5 邊:",
      [(data.edge_index[0,i].item(), data.edge_type[i].item(),
        data.edge_index[1,i].item()) for i in range(5)])

    torch.save(data, out_path)
    print(f"Graph → {out_path}  |  Nodes: {data.num_nodes:,}  Edges: {data.edge_index.size(1):,}")
    print(f"◉ Train 標籤數：{data.train_mask.sum().item()}")
    print(f"◉ Val   標籤數：{data.val_mask.sum().item()}")
    print(f"◉ Test  標籤數：{data.test_mask.sum().item()}")

if __name__ == "__main__":
    main()

