import collections, math

def build_time_dict(triples, half_life=3):
    """
    同一病人依 visit_date 排序：
      visit_idx = 0,1,...
      decay_w   = 0.5 ** (visit_idx/half_life)
    回傳 dict[id(triple)] = (visit_idx, decay_w)
    """
    visits = collections.defaultdict(list)
    for t in triples:
        head = t["head"]
        # 若 head 裡有 ":"，則拆分取得 id；否則直接使用 head 當 pid
        pid = head.split(":", 1)[-1]
        visits[pid].append(t)

    time_info = {}
    for pid, lst in visits.items():
        lst.sort(key=lambda x: x["visit_date"])
        for idx, t in enumerate(lst):
            decay_w = 0.5 ** (idx / half_life)
            time_info[id(t)] = (idx, decay_w)
    return time_info

