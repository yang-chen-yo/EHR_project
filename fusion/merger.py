from pathlib import Path
import os
import json
import re
import time
from typing import List, Dict
from datetime import datetime
from tqdm import tqdm

from config import K_UMLS, K_PUBMED, PUBMED_API_KEY
from embed.encoder_google import Encoder
from embed.faiss_index import FaissIndex
from retrieval.pubmed_client import PubMedClient
from retrieval.umls_client import UMLSClient
from fusion.scoring import cosine_similarity, score_pubmed_hit
from kg.triple import Triple
from fusion.triple_generation_hf import generate_triples_via_gemma

# Global encoder (constructed once)
encoder = Encoder()

def fuse_and_score_progress(
    patient_id: str,
    visit_id: str,
    patient_text: str,
    umls_dir: str,
    pubmed_email: str,
    patient_fields: Dict[str, list],
    pbar: tqdm | None = None,
    k_umls: int = K_UMLS,
    k_pubmed: int = K_PUBMED,
) -> Dict[str, List]:
    """
    embed → UMLS → PubMed retrieval with caching; returns {'umls':…, 'pubmed':…}
    """
    cache_dir = Path("retrieval_cache") / patient_id / visit_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    umls_fp = cache_dir / "umls.json"
    pubmed_fp = cache_dir / "pubmed.json"

    if umls_fp.exists() and pubmed_fp.exists():
        umls_hits = json.load(open(umls_fp, encoding="utf-8"))
        pubmed_hits = json.load(open(pubmed_fp, encoding="utf-8"))
    else:
        # (1) Embed
        if pbar: pbar.set_description("embed")
        qvec = encoder.encode([patient_text])[0]

        # (2) UMLS
        if pbar: pbar.set_description("umls")
        client_u = UMLSClient(umls_dir)
        cuis = client_u.concepts
        names = [client_u.concept_names[c] for c in cuis]
        name_vecs = encoder.encode(names)
        idx_u = FaissIndex(name_vecs.shape[1])
        idx_u.build(name_vecs)
        ids_u, sims_u = idx_u.search(qvec, k_umls)
        umls_hits = [
            {"cui": cuis[i], "name": names[i], "score": float(sims_u[j])}
            for j, i in enumerate(ids_u)
        ]

        # (3) PubMed
        if pbar: pbar.set_description("pubmed")
        client_p = PubMedClient(email=pubmed_email, api_key=PUBMED_API_KEY)
        topics = [
            c["name"]
            for fld in ("conditions", "procedures", "drugs")
            for sub in patient_fields.get(fld, [])
            for c in sub
        ]
        all_pmids = []
        for name in topics:
            all_pmids += client_p.search(name, retmax=k_pubmed)
        all_pmids = list(dict.fromkeys(all_pmids))[: k_pubmed * max(len(topics), 1)]
        arts = client_p.fetch_abstracts(all_pmids)

        abstracts = [art["abstract"] for art in arts]
        abstract_vecs = encoder.encode(abstracts)

        pubmed_hits = []
        for art, vec in zip(arts, abstract_vecs):
            # ① 相似度計算
            sim   = cosine_similarity(qvec, vec)
            score = score_pubmed_hit(
                sim,
                art.get("year") or datetime.now().year
            )

            # ② 時間戳：優先用 fetch_abstracts 解析到的 timestamp
            ts = art.get("timestamp")
            if not ts:
                yr = art.get("year")
                if yr and str(yr).isdigit():    # 只有年份 → 補成 YYYY-01-01
                    ts = f"{yr}-01-01"

            # ③ 收進 pubmed_hits（注意，這行前面不要多縮排）
            pubmed_hits.append({
                **art,
                "score": score,
                "timestamp": ts          # ← 這裡就帶進去了
            })
        pubmed_hits.sort(key=lambda x: x['score'], reverse=True)

        # Cache to disk
        json.dump(umls_hits, open(umls_fp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(pubmed_hits, open(pubmed_fp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    if pbar: pbar.set_description("done")
    return {"umls": umls_hits, "pubmed": pubmed_hits}


def merge_to_triples(
    patient_id: str,
    visit_id: str,
    fused: Dict[str, List],
    patient_context: str,
    patient_fields: Dict[str, list],
    visit_date: str | None = None,
    output_dir: str = "triples_output",
    accumulate: bool = True,
) -> List[Triple]:
    """
    只用 UMLS + RAG 三元組，並支援快取與中斷續跑
    """
    # 1) 準備輸出與快取資料夾
    pdir      = Path(output_dir) / patient_id
    pdir.mkdir(parents=True, exist_ok=True)
    out_path  = pdir / f"triples_{visit_id}.json"

    cache_dir = Path("retrieval_cache") / patient_id / visit_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    last_fp   = cache_dir / "last_batch.txt"
    start_i   = int(last_fp.read_text()) if last_fp.exists() else 0

    triples: List[Triple] = []

    # ① UMLS 三元組
    umls_hits = fused["umls"]
    for hit in umls_hits:
        triples.append(Triple(
            head=patient_id, head_type="Patient",
            relation="HAS_DISEASE",
            tail=hit["name"], tail_type="Disease",
            visit_id=visit_id, visit_date=visit_date,
            timestamp=visit_date,
            source="UMLS",
            weight=hit["score"]
        ))

    # ② RAG via Gemma-3
    pubmed_hits = fused["pubmed"]
    abstracts   = [h["abstract"] for h in pubmed_hits]
    abs2ts      = {h["abstract"]: h.get("timestamp") for h in pubmed_hits}
    best_score  = pubmed_hits[0]["score"] if pubmed_hits else None

    batch_size = 1
    for i in range(start_i, len(abstracts), batch_size):
        batch_idx = i // batch_size
        cache_fp  = cache_dir / f"rag_{batch_idx}.json"
        batch     = abstracts[i:i+batch_size]

        # 2.1 快取檢查
        if cache_fp.exists():
            rag = json.load(open(cache_fp, "r", encoding="utf-8"))
        else:
            # 2.2 呼叫模型＋快取＋紀錄進度
            try:
                rag = generate_triples_via_gemma(
                    patient_context=patient_context,
                    abstracts=batch,
                    batch_size=batch_size,
                    max_output_tokens=1024,
                    patient_id=patient_id
                )
            except Exception as e:
                print(f"[ERROR] RAG batch {batch_idx} 失敗：{e}")
                break
            json.dump(rag, open(cache_fp, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            last_fp.write_text(str(i))

        # 2.3 解析並補齊欄位
        for t in rag:
            if not all(k in t for k in ("head", "head_type", "relation", "tail", "tail_type")):
                continue

            # ── ① 先拿 timestamp：t 自帶 → abs2ts → 最後才 fallback visit_date
            ts = t.get("timestamp") or abs2ts.get(batch[0]) or visit_date

            # ── ② 如果是 int/only year，補成 YYYY-01-01
            if ts and isinstance(ts, int):
                ts = f"{ts}-01-01"
            elif ts and isinstance(ts, str) and re.fullmatch(r"\d{4}", ts):
                ts = f"{ts}-01-01"

            # ── ③ 最後確保是字串；若仍為 None 則給空字串
            ts = str(ts) if ts is not None else ""

            triples.append(
                Triple(
                    head=t.get("head", patient_id),
                    head_type=t.get("head_type", "Patient"),
                    relation=t.get("relation", "UnknownRelation"),
                    tail=t.get("tail", ""),
                    tail_type=t.get("tail_type", "Unknown"),
                    visit_id=visit_id,
                    visit_date=visit_date,
                    timestamp=ts,
                    source="RAG",
                    weight=best_score,
                )
            )
    # ③ 去重＆寫檔
    if accumulate:
        existing = json.load(open(out_path, "r", encoding="utf-8")) if out_path.exists() else []
        merged   = {json.dumps(x, sort_keys=True) for x in existing + [t.__dict__ for t in triples]}
        json.dump([json.loads(s) for s in merged],
                  open(out_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"[DEBUG] 已寫入 {out_path}，共 {len(merged)} 條三元組")

    return triples
