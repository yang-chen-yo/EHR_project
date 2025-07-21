import pickle, torch
from sklearn.model_selection import train_test_split
from data.triple_loader import norm_patient

def load_labels(pkl_path, ent2id, task):
    samples = pickle.load(open(pkl_path, "rb"))
    N = len(ent2id)
    # 初始化 y
    if task == "drugrec":
        out_dim = samples[0]["label"]["drugs_ind"].shape[0]
        y = torch.zeros((N, out_dim), dtype=torch.float)
    else:
        y = torch.full((N,), -1., dtype=torch.float)

    # 填入標籤
    for s in samples:
        pid = norm_patient(str(s["patient_id"]))
        if pid not in ent2id: continue
        nid = ent2id[pid]
        if task == "drugrec":
            y[nid] = torch.tensor(s["label"]["drugs_ind"], dtype=torch.float)
        elif task == "mortality":
            lab = s["label"]
            val = lab["mortality"] if isinstance(lab, dict) else lab
            y[nid] = float(val)
        elif task == "readmission":
            lab = s["label"]
            val = lab["readmit_30d"] if isinstance(lab, dict) else lab
            y[nid] = float(val)
        else:  # lenofstay
            lab = s["label"]
            val = lab["los_bucket"] if isinstance(lab, dict) else lab
            y[nid] = int(val)

    # 收集有標籤節點
    idx = torch.nonzero(y != -1, as_tuple=False).view(-1).tolist()
    labels = [y[i].item() for i in idx]
    if not idx:
        raise RuntimeError("❌ 無任何標籤對應到圖中節點，請檢查命名規則！")

    # 1) 切 train + temp
    stratify = labels if len(set(labels)) > 1 else None
    train_idx, temp_idx, _, temp_labels = train_test_split(
        idx, labels,
        test_size=0.3,
        stratify=stratify,
        random_state=42
    )

    # 2) 切 temp → val / test (只拆 index)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=temp_labels if stratify is not None else None,
        random_state=42
    )

    # 建立 mask
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask   = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    return y, dict(
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

