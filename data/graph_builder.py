import torch
from torch_geometric.data import Data
from .triple_loader import load_triples
from .triple_loader import build_maps
from .time_utils import build_time_dict
from .label_loader import load_labels

def build_graph(json_dir, pkl_path, task):
    triples    = load_triples(json_dir)
    ent2id, rel2id = build_maps(triples)
    time_info  = build_time_dict(triples)

    h  = [ent2id[t["head"]] for t in triples]
    t_ = [ent2id[t["tail"]] for t in triples]
    r  = [rel2id[t["relation"]] for t in triples]
    w  = [time_info[id(t)][1] for t in triples]          # decay_w

    edge_index  = torch.tensor([h, t_], dtype=torch.long)
    edge_type   = torch.tensor(r, dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float).unsqueeze(-1)  # [E,1]

    y, masks    = load_labels(pkl_path, ent2id, task)

    data = Data(x=None,
                edge_index=edge_index,
                edge_type=edge_type,
                edge_weight=edge_weight,
                y=y,
                **masks)
    data.num_nodes     = len(ent2id)
    data.num_relations = len(rel2id)
    return data

