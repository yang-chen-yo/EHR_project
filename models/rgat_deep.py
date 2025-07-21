import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import RGATConv, global_mean_pool  

class DeepRGAT(nn.Module):
    """
    多層 Relational GAT
      - num_layers  : Conv 層數（>=1）
      - heads       : Attention 頭數
      - edge_dim    : 是否使用 edge_weight (時間衰減)；若不用傳 0
      - out_dim     : 任務輸出維度
    """
    def __init__(self, num_nodes, num_rel, hid=64,
                 num_layers=4, heads=4, edge_dim=0, out_dim=1, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, hid)

        self.layers = nn.ModuleList()
        in_ch = hid
        for _ in range(num_layers):
            self.layers.append(
                RGATConv(in_ch, hid,
                         num_relations=num_rel,
                         heads=heads,
                         edge_dim=edge_dim,
                         dropout=dropout)
            )
            # 下一層輸入維度 = hid*heads（因 RGATConv 會 concat）
            in_ch = hid * heads

        # node‑level 任務 → 線性映射到 out_dim
        self.lin = nn.Linear(in_ch, out_dim)

    def forward(self, edge_index, edge_type, edge_weight=None, batch=None, x_idx=None):
        # 若 patient_mode，x_idx 會是 batch.x (node indices)，otherwise 用全圖 embedding
        if batch is None:
            x = self.emb.weight
        else:
            x = self.emb(x_idx)                  # 貼上子圖的 node index
        for conv in self.layers:
            x = F.elu(conv(x, edge_index, edge_type, edge_weight))
        out = self.lin(x)
        return global_mean_pool(out, batch) if batch is not None else out
