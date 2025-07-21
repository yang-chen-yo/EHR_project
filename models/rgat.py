import torch.nn as nn, torch
from torch_geometric.nn import RGATConv
import torch.nn.functional as F

class RGAT(nn.Module):
    def __init__(self, num_nodes, num_rel, hid=256, out_dim=1):
        super().__init__()
        self.emb  = nn.Embedding(num_nodes, hid)
        self.conv = RGATConv(hid, hid, num_rel, heads=8)   # 無 edge_dim
        self.lin  = nn.Linear(hid, out_dim)

    def forward(self, edge_index, edge_type):
        x = self.emb.weight
        x = F.elu(self.conv(x, edge_index, edge_type))
        return self.lin(x)

