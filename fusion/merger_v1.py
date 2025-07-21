# fusion/merger.py
from typing import List, Dict
from datetime import datetime
import os, json

from config import K_UMLS, K_PUBMED, PUBMED_API_KEY
from embed.encoder import Encoder
from embed.faiss_index import FaissIndex
from retrieval.pubmed_client import PubMedClient
from retrieval.umls_client import UMLSClient
from fusion.scoring import cosine_similarity, score_pubmed_hit
from fusion.triple_generation_hf import generate_triples_local_llama2
from kg.triple import Triple
from tqdm import tqdm

# 全域只初始化一次 Encoder
encoder = Encoder()


def fuse_and_score_progress(
    patient_text: str,
    umls_dir: str,
    pubmed_email: str,
    patient_fields: Dict[str, list],
    pbar: tqdm | None = None,          # ← 傳入外層 tqdm 物件
    k_umls: int = K_UMLS,
    k_pubmed: int = K_PUBMED,
) -> Dict[str, List]:
    """
    與 fuse_and_score 行為相同，但可在同一條 pbar 上顯示階段進度：
      embed → umls → pubmed → done
    """
    # ---------- ① Patient embed ----------
    if pbar: pbar.set_description("embed")
    qvec = encoder.encode([patient_text])[0]

    # ---------- ② UMLS ----------
    if pbar: pbar.set_description("umls")
    client_u = UMLSClient(umls_dir)
    cuis  = client_u.concepts
    names = [client_u.concept_names[c] for c in cuis]
    name_vecs = encoder.encode(names)
    idx_u = FaissIndex(name_vecs.shape[1])
    idx_u.build(name_vecs)
    ids_u, sims_u = idx_u.search(qvec, k_umls)
    umls_hits = [
        {"cui": cuis[i], "name": names[i], "score": float(sims_u[j])}
        for j, i in enumerate(ids_u)
    ]

    # ---------- ③ PubMed ----------
    if pbar: pbar.set_description("pubmed")
    client_p = PubMedClient(email=pubmed_email,api_key=PUBMED_API_KEY)
    topics = [
        c["name"]
        for field in ("conditions", "procedures", "drugs")
        for sub in patient_fields.get(field, [])
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
        sim   = cosine_similarity(qvec, vec)
        score = score_pubmed_hit(sim, art.get("year") or datetime.now().year)
        pubmed_hits.append({**art, "score": score})
    pubmed_hits.sort(key=lambda x: x["score"], reverse=True)

    if pbar: pbar.set_description("done")
    return {"umls": umls_hits, "pubmed": pubmed_hits}

# ---------- 下面是 Triple 合併 ----------
def _escape_braces(text: str) -> str:
    """將字串中的單層 { } 轉成 {{ }} 以避免 .format() 出錯"""
    return text.replace("{", "{{").replace("}", "}}")


def merge_to_triples(
    patient_id: str,
    fused: Dict[str, List],
    patient_context: str,
    patient_fields: Dict[str, list],
    output_dir: str = 'triples_output',
    accumulate: bool = True,
) -> List[Triple]:
    os.makedirs(os.path.join(output_dir, patient_id), exist_ok=True)
    triples: List[Triple] = []

    # --------- Debug 基本資訊 ----------
    print(f"\n[DEBUG] === Start merge for PatientID={patient_id} ===")
    print(f"[DEBUG] → Context length: {len(patient_context)}")
    print(f"[DEBUG] → PubMed hits: {len(fused['pubmed'])}, UMLS hits: {len(fused['umls'])}")
    topics = [
        c["name"]
        for field in ("conditions", "procedures", "drugs")
        for sub in patient_fields.get(field, [])
        for c in sub
    ]
    print(f"[DEBUG] → Query topics: {topics}")

    # --------- UMLS triples ----------
    for hit in fused['umls']:
        triples.append(
            Triple(
                head=patient_id, head_type='Patient',
                relation='HAS_DISEASE', tail=hit['cui'], tail_type='Disease',
                source='UMLS', weight=hit['score']
            )
        )

    # --------- RAG triples ----------
    best_score = fused['pubmed'][0]['score'] if fused['pubmed'] else None
    abstracts = [h['abstract'] for h in fused['pubmed']]

    print(f"[DEBUG] → Generating triples from {len(abstracts)} abstracts")
    if abstracts:
        print(f"[DEBUG] → First abstract preview: {abstracts[0][:120]}...")

    try:
        rag_list = generate_triples_local_llama2(
            patient_context=_escape_braces(patient_context),
            abstracts=[_escape_braces(a) for a in abstracts],
            umls_facts=None
        )
    except Exception as e:
        print(f"[ERROR] Triple generation failed for patient {patient_id}: {e}")
        return []

    print(f"[DEBUG] → Generated {len(rag_list)} RAG triples")

    for item in rag_list:
        triples.append(
            Triple(
                head=item['head'], head_type=item['head_type'],
                relation=item['relation'], tail=item['tail'], tail_type=item['tail_type'],
                timestamp=item.get('timestamp'), source='PubMed', weight=best_score
            )
        )
        if accumulate:
            pdir = os.path.join(output_dir, patient_id)
            fname = f"{item['relation']}_{item['tail']}.json"
            with open(os.path.join(pdir, fname), 'w', encoding='utf-8') as f:
                json.dump(item, f, ensure_ascii=False, indent=2)

    print(f"[DEBUG] === Done for PatientID={patient_id} ===")
    return triples if accumulate else []

