#!/usr/bin/env python3
# coding: utf-8
import torch, argparse
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from config.tasks import TASKS
from config.hparams import HPARAMS

def load_model(task, use_time, device):
    from models.rgat_time import RGAT_Time
    from models.rgat      import RGAT
    Model = RGAT_Time if use_time else RGAT
    ckpt = torch.load(f"checkpoints/{task}_{'time' if use_time else 'notime'}.pt",
                      map_location=device)
    model = Model(ckpt["model"]["emb.weight"].size(0),  # num_nodes
                  ckpt["model"]["conv.rel_att"].size(0),# num_relations
                  hid=HPARAMS[task]["hid"],
                  out_dim=1).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--task", required=True, choices=TASKS)
    pa.add_argument("--graph_pt", required=True)
    pa.add_argument("--device", default="cuda")
    pa.add_argument("--use_time", action="store_true")
    args = pa.parse_args()

    data  = torch.load(args.graph_pt, map_location=args.device).to(args.device)
    model = load_model(args.task, args.use_time, args.device)

    with torch.no_grad():
        logits = (model(data.edge_index, data.edge_type, data.edge_weight)
                  if args.use_time else
                  model(data.edge_index, data.edge_type))
    mask = data.test_mask
    if args.task == "lenofstay":
        pred_cls = logits[mask].argmax(-1).cpu()
        y_true   = data.y[mask].cpu()
        print("Accuracy:", accuracy_score(y_true, pred_cls))
        print("F1‑macro:", f1_score(y_true, pred_cls, average="macro"))
    else:
        prob  = logits[mask].sigmoid().squeeze().cpu()
        y_true= data.y[mask].cpu()
        print("AUROC:", roc_auc_score(y_true, prob))
        print("F1:",    f1_score(y_true, (prob>=0.5)))

if __name__ == "__main__":
    main()

