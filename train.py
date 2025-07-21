#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py —— Patient‑subgraph GNN 訓練腳本（含 Bootstrap Oversample + Dup Eval）
任務：mortality / readmission / lenofstay / drugrec
功能：70‑15‑15 split 或 Stratified‑K‑fold、Bootstrap oversample
輸出：checkpoints/best.pt, loss_curve.png、acc_curve.png（平滑）、classification_report.txt
"""

import argparse, warnings, random
from pathlib import Path
import numpy as np
import torch, matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, cohen_kappa_score,
    classification_report
)
import torch.nn as nn

import config
from data.patient_dataset import build_patient_dataset
from models.rgat_deep import DeepRGAT

warnings.filterwarnings("ignore", message="An issue occurred while importing 'torch-sparse'")

# ──────────────── 小工具 ────────────────
def scalar_label(y):
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    if hasattr(y, "__len__") and len(y) > 1:
        return int(np.argmax(y))
    return int(y)

def moving_avg(x, w=5):
    return np.convolve(x, np.ones(w) / w, mode="same")

def dup_indices(idx, labels, times=1):
    """把 idx 中的正例複製 times 倍（保持順序）"""
    pos = idx[labels[idx] == 1]
    if len(pos) == 0 or times <= 1:
        return idx
    extra = np.repeat(pos, times - 1)
    return np.concatenate([idx, extra])

# ──────────────── 主程式 ────────────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--task", required=True, choices=config.TASKS)
    pa.add_argument("--json_dir", default="triples_output")
    pa.add_argument("--device", default="cuda")
    pa.add_argument("--use_time", action="store_true")

    # Hyperparameters
    pa.add_argument("--batch_size", type=int, default=32)
    pa.add_argument("--epochs", type=int, default=60)
    pa.add_argument("--hid", type=int, default=64)
    pa.add_argument("--layers", type=int, default=4)
    pa.add_argument("--heads", type=int, default=4)
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--dropout", type=float, default=0.5)
    pa.add_argument("--patience", type=int, default=12)

    # Oversample + Dup Eval
    pa.add_argument("--oversample", action="store_true",
                    help="對正例做 bootstrap oversample")
    pa.add_argument("--boot_mult", type=int, default=1,
                    help="bootstrap 倍數 (train)")
    pa.add_argument("--dup_eval", action="store_true",
                    help="對 val/test 的正例也重複 oversample")

    # Split
    pa.add_argument("--kfold", type=int, default=0,
                    help="0/1 隨機 70/15/15；>=2 啟用 Stratified KFold")
    pa.add_argument("--fold", type=int, default=0,
                    help="選第幾折 (0-based)")
    args = pa.parse_args()

    # 建立 checkpoint 資料夾
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42); np.random.seed(42); random.seed(42)

    # Load dataset
    conf = config.TASKS[args.task]
    dataset, ent2id, rel2id, out_dim = build_patient_dataset(
        args.json_dir, conf["pkl"], args.task, use_time=args.use_time
    )
    mode = {"mortality":"binary","readmission":"binary",
            "lenofstay":"multiclass","drugrec":"multilabel"}[args.task]

    labels = np.array([scalar_label(d.y) for d in dataset])
    idx = np.arange(len(labels))

    # Split
    if args.kfold >= 2:
        skf = StratifiedKFold(args.kfold, shuffle=True, random_state=42)
        splits = list(skf.split(idx, labels))
        tr_all, test_idx = splits[args.fold % args.kfold]
        _, val_idx = splits[(args.fold+1) % args.kfold]
        train_idx = np.setdiff1d(tr_all, test_idx)
    else:
        train_idx, test_idx = train_test_split(idx, test_size=0.15,
                                               stratify=labels, random_state=42)
        train_idx, val_idx  = train_test_split(train_idx, test_size=0.1765,
                                               stratify=labels[train_idx], random_state=42)

    # dup_eval for binary
    if mode == "binary" and args.dup_eval:
        val_idx  = dup_indices(val_idx, labels, args.boot_mult)
        test_idx = dup_indices(test_idx, labels, args.boot_mult)

    train_ds = [dataset[i] for i in train_idx]
    val_ds   = [dataset[i] for i in val_idx]
    test_ds  = [dataset[i] for i in test_idx]
    print(f"◎ Split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    # Oversample sampler (support binary & multiclass)
    sampler = None
    if args.oversample:
        y_tr = np.array([scalar_label(d.y) for d in train_ds])
        if mode == "binary":
            neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
            weights = np.where(y_tr==1, neg/pos, 1.0)
        else:
            counts = np.bincount(y_tr, minlength=(10 if mode=="multiclass" else out_dim))
            weights = counts.sum() / (counts[y_tr] + 1e-8)
        weights = weights.astype(np.float32)
        sampler = WeightedRandomSampler(
            torch.tensor(weights),
            num_samples=len(weights) * args.boot_mult,
            replacement=True
        )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=sampler, shuffle=(sampler is None), drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader= DataLoader(test_ds,batch_size=128, shuffle=False)

    # Model
    edge_dim = 1 if args.use_time else 0
    out_dim_task = 10 if mode=="multiclass" else out_dim
    model = DeepRGAT(len(ent2id), len(rel2id),
                     hid=args.hid, num_layers=args.layers, heads=args.heads,
                     edge_dim=edge_dim, out_dim=out_dim_task,
                     dropout=args.dropout).to(device)

    # Loss & Metric
    if mode == "binary":
        pos_w = (labels[train_idx]==0).sum()/max((labels[train_idx]==1).sum(),1)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=device))
        def metric(out, truth):
            prob = out.sigmoid().view(-1).cpu().numpy()
            y    = truth.view(-1).cpu().numpy()
            return accuracy_score(y,(prob>=0.5).astype(int)), roc_auc_score(y,prob)

    elif mode == "multiclass":
        cnts = np.bincount(labels[train_idx], minlength=out_dim_task)
        weights = torch.tensor(1/np.clip(cnts,1,None), dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        def metric(out, truth):
            pred = out.softmax(-1).argmax(-1).cpu().numpy()
            y    = truth.cpu().numpy()
            return accuracy_score(y, pred), cohen_kappa_score(y, pred, weights='quadratic')

    else:  # multilabel
        loss_fn = nn.BCEWithLogitsLoss()
        def metric(out, truth):
            prob = out.sigmoid().cpu().numpy()
            y    = truth.cpu().numpy()
            pred = (prob>=0.5).astype(int)
            return accuracy_score(y, pred), f1_score(y, pred, average='samples')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler    = GradScaler()

    # Train Loop
    hist = {'ep':[], 'loss':[], 'tr_acc':[], 'val_acc':[]}
    best_val, patience = 0.0, 0
    for ep in range(1, args.epochs+1):
        model.train(); total_loss = 0
        for batch in tqdm(train_loader, desc=f"E{ep:02d}", leave=False):
            batch = batch.to(device); optimizer.zero_grad()
            with autocast(device_type=device.type):
                out = model(batch.edge_index, batch.edge_type,
                            batch.edge_weight if args.use_time else None,
                            batch.batch, batch.x)
                if mode=="multiclass":
                    y = batch.y
                    if y.dim()>1: y = y.argmax(dim=1)
                    loss = loss_fn(out, y.long())
                else:
                    loss = loss_fn(out.squeeze(), batch.y.float().view(-1))
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total_loss += loss.item()
        avg_loss = total_loss/len(train_loader)

        # Train metric on last batch
        tr_acc, _ = metric(out.detach(), batch.y.detach())

        # Val
        model.eval(); outs, ys = [], []
        with torch.no_grad(), autocast(device_type=device.type):
            for vb in val_loader:
                vb = vb.to(device)
                o = model(vb.edge_index, vb.edge_type,
                          vb.edge_weight if args.use_time else None,
                          vb.batch, vb.x)
                outs.append(o); ys.append(vb.y)
        val_out = torch.cat(outs); val_y = torch.cat(ys)
        val_acc, _ = metric(val_out, val_y)

        hist['ep'].append(ep); hist['loss'].append(avg_loss)
        hist['tr_acc'].append(tr_acc); hist['val_acc'].append(val_acc)
        print(f"E{ep:02d} loss={avg_loss:.3f} TrACC={tr_acc:.3f} ValACC={val_acc:.3f}")

        if val_acc > best_val:
            best_val, patience = val_acc, 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            patience += 1
            if patience >= args.patience:
                print("Early stopping!"); break

    # Test
    model.load_state_dict(torch.load(ckpt_dir / "best.pt"))
    model.eval(); outs, ys = [], []
    with torch.no_grad(), autocast(device_type=device.type):
        for tb in test_loader:
            tb = tb.to(device)
            o = model(tb.edge_index, tb.edge_type,
                      tb.edge_weight if args.use_time else None,
                      tb.batch, tb.x)
            outs.append(o); ys.append(tb.y)
    test_out = torch.cat(outs); test_y = torch.cat(ys)
    test_acc, _ = metric(test_out, test_y)
    print(f"★ Test ACC={test_acc:.3f}")

    # Classification report
    if mode == "multiclass":
        preds = test_out.softmax(-1).argmax(-1).cpu().numpy()
        truths= test_y.cpu().numpy()
    elif mode == "binary":
        preds = (test_out.sigmoid().view(-1).cpu().numpy()>=0.5).astype(int)
        truths= test_y.view(-1).cpu().numpy().astype(int)
    else:
        preds = (test_out.sigmoid().cpu().numpy()>=0.5).astype(int)
        truths= test_y.cpu().numpy().astype(int)
    report = classification_report(truths, preds, digits=4)
    print(report)
    (ckpt_dir / "classification_report.txt").write_text(report)

    # Plot curves
    plt.figure(); plt.plot(hist['ep'], moving_avg(hist['loss']), '-o'); plt.xlabel('Epoch'); plt.ylabel('Train Loss'); plt.tight_layout(); plt.savefig(ckpt_dir / 'loss_curve.png'); plt.close()
    plt.figure(); plt.plot(hist['ep'], moving_avg(hist['tr_acc']), '-o', label='Train ACC'); plt.plot(hist['ep'], moving_avg(hist['val_acc']), '-s', label='Val ACC'); plt.xlabel('Epoch'); plt.ylabel('ACC'); plt.legend(); plt.tight_layout(); plt.savefig(ckpt_dir / 'acc_curve.png')

if __name__ == "__main__":
    main()

