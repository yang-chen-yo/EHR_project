# fusion/merger.py
"""Fuse embeddings, UMLS & PubMed retrieval, and merge triples via GitHub GPT."""

import os
import json
from typing import List, Dict
from datetime import datetime
from tqdm import tqdm

from config import K_UMLS, K_PUBMED, PUBMED_API_KEY
from embed.encoder import Encoder
from embed.faiss_index import FaissIndex
from retrieval.pubmed_client import PubMedClient
from retrieval.umls_client import UMLSClient
from fusion.scoring import cosine_similarity, score_pubmed_hit
from kg.triple import Triple
from fusion.triple_generation_hf import generate_triples_via_github   # RAG 函式

# -------- global encoder (只建一次) ----------
encoder = Encoder()
# --------------------------------------------


# ---------- 1. 取得 UMLS / PubMed hit ----------
def fuse_and_score_progress(
    patient_text: str,
    umls_dir: str,
    pubmed_email: str,
    patient_fields: Dict[str, list],
    pbar: tqdm | None = None,
    k_umls: int = K_UMLS,
    k_pubmed: int = K_PUBMED,
) -> Dict[str, List]:
    """embed → UMLS → PubMed 三階段檢索；回傳 {'umls': …, 'pubmed': …}"""

    # (1) 句向量
    if pbar: pbar.set_description("embed")
    qvec = encoder.encode([patient_text])[0]

    # (2) UMLS
    if pbar: pbar.set_description("umls")
    client_u = UMLSClient(umls_dir)
    cuis      = client_u.concepts
    names     = [client_u.concept_names[c] for c in cuis]
    name_vecs = encoder.encode(names)
    idx_u     = FaissIndex(name_vecs.shape[1])
    idx_u.build(name_vecs)
    ids_u, sims_u = idx_u.search(qvec, k_umls)
    umls_hits = [{"cui": cuis[i], "name": names[i], "score": float(sims_u[j])}
                 for j, i in enumerate(ids_u)]

    # (3) PubMed
    if pbar: pbar.set_description("pubmed")
    client_p = PubMedClient(email=pubmed_email, api_key=PUBMED_API_KEY)
    topics   = [c["name"]
                for fld in ("conditions", "procedures", "drugs")
                for sub in patient_fields.get(fld, [])
                for c   in sub]
    all_pmids = []
    for name in topics:
        all_pmids += client_p.search(name, retmax=k_pubmed)
    all_pmids = list(dict.fromkeys(all_pmids))[: k_pubmed * max(len(topics), 1)]
    arts      = client_p.fetch_abstracts(all_pmids)

    abstracts     = [art["abstract"] for art in arts]
    abstract_vecs = encoder.encode(abstracts)

    pubmed_hits = []
    for art, vec in zip(arts, abstract_vecs):
        sim   = cosine_similarity(qvec, vec)
        score = score_pubmed_hit(sim, art.get("year") or datetime.now().year)
        pubmed_hits.append({**art, "score": score})
    pubmed_hits.sort(key=lambda x: x["score"], reverse=True)

    if pbar: pbar.set_description("done")
    return {"umls": umls_hits, "pubmed": pubmed_hits}


# ---------- 2. 合併並落檔 ----------
def merge_to_triples(
    patient_id: str,
    fused: Dict[str, List],
    patient_context: str,
    patient_fields: Dict[str, list],
    visit_date: str | None = None,                      # ← 新增參數
    output_dir: str = "triples_output",
    accumulate: bool = True,
) -> List[Triple]:
    """
    Merge UMLS + RAG triples.
    - 每 2 篇摘要呼叫一次 GPT。
    - accumulate=True 時僅輸出 combined_triples.json。
    """
    pdir = os.path.join(output_dir, patient_id)
    os.makedirs(pdir, exist_ok=True)
    triples: List[Triple] = []

    # ① UMLS
    for hit in fused["umls"]:
        triples.append(Triple(
            head=patient_id, head_type="Patient",
            relation="HAS_DISEASE",
            tail=hit["cui"], tail_type="Disease",
            visit_date=visit_date,          # <<<<<<
            source="UMLS",  weight=hit["score"]
        ))

    # ② RAG  (batch=2)
    abstracts  = [h["abstract"] for h in fused["pubmed"]]
    best_score = fused["pubmed"][0]["score"] if fused["pubmed"] else None
    for i in range(0, len(abstracts), 2):
        batch = abstracts[i:i+2]
        try:
            rag = generate_triples_via_github(patient_context, batch, max_new_tokens=1000)
            for item in rag:
                triples.append(Triple(
                    head=item["head"], head_type=item["head_type"],
                    relation=item["relation"],
                    tail=item["tail"], tail_type=item["tail_type"],
                    visit_date=visit_date,                # <<<<<<
                    timestamp=item.get("timestamp"),
                    source=item.get("source", "PubMed"),
                    weight=best_score
                ))
        except Exception as e:
            print(f"[WARN] GPT batch {i//2+1} failed:", e)

    # ③ 只寫一檔
    if accumulate:
        with open(os.path.join(pdir, "combined_triples.json"), "w", encoding="utf-8") as f:
            json.dump([t.__dict__ for t in triples], f, ensure_ascii=False, indent=2)

    print(f"[DEBUG] Done for {patient_id} – {len(triples)} triples")
    return triples
