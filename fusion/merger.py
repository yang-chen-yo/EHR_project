#fusion/merger.py
from typing import List, Dict
from datetime import datetime

from config import K_UMLS, K_PUBMED
from embed.encoder import Encoder
from embed.faiss_index import FaissIndex
from retrieval.pubmed_client import PubMedClient
from retrieval.umls_client import UMLSClient
from fusion.scoring import cosine_similarity, score_pubmed_hit
from fusion.triple_generation_hf import generate_triples_local_llama2
from kg.triple import Triple


def fuse_and_score(
    patient_text: str,
    umls_dir: str,
    pubmed_email: str,
    k_umls: int = K_UMLS,
    k_pubmed: int = K_PUBMED,
) -> Dict[str, List]:
    """Vector‐based retrieval on UMLS and PubMed.

    Returns a dict with keys:
        'umls':   List[{'cui','name','score'}],
        'pubmed': List[{'pmid','title','abstract','score','year'}],
        'umls_facts': List[str]
    """
    encoder = Encoder()
    qvec = encoder.encode([patient_text])[0]

    # ── UMLS retrieval ───────────
    client_u = UMLSClient(umls_dir)
    cuis = client_u.concepts
    names = [client_u.concept_names[c] for c in cuis]
    name_vecs = encoder.encode(names)

    idx_u = FaissIndex(name_vecs.shape[1])
    idx_u.build(name_vecs)
    ids_u, sims_u = idx_u.search(qvec, k_umls)

    umls_hits = [
        {'cui': cuis[i], 'name': names[i], 'score': float(sims_u[j])}
        for j, i in enumerate(ids_u)
    ]

    # Build plain-text UMLS facts (max 2 per hit)
    umls_facts: List[str] = []
    for hit in umls_hits:
        cui = hit['cui']
        facts = [t for t in client_u.relation_triples if t['cui1'] == cui or t['cui2'] == cui][:2]
        for t in facts:
            h_name = client_u.concept_names.get(t['cui1'], t['cui1'])
            rel = t['relation']
            t_name = client_u.concept_names.get(t['cui2'], t['cui2'])
            umls_facts.append(f"{h_name} {rel} {t_name}")

    # ── PubMed retrieval ─────────
    client_p = PubMedClient(email=pubmed_email)
    pmids = client_p.search(patient_text, retmax=k_pubmed)
    arts = client_p.fetch_abstracts(pmids)

    pubmed_hits = []
    for art in arts:
        vec = encoder.encode([art['abstract']])[0]
        sim = cosine_similarity(qvec, vec)
        year = art.get('year') or datetime.now().year
        score = score_pubmed_hit(sim, year)
        pubmed_hits.append({**art, 'score': score})

    pubmed_hits.sort(key=lambda x: x['score'], reverse=True)

    return {
        'umls': umls_hits,
        'pubmed': pubmed_hits,
        'umls_facts': umls_facts,
    }


def merge_to_triples(
    patient_id: str,
    fused: Dict[str, List],
    patient_context: str,
) -> List[Triple]:
    """Combine EHR‐derived UMLS triples with RAG‐derived PubMed triples."""

    triples: List[Triple] = []

    # 1) UMLS → HAS_DISEASE  --------------------------------------------
    seen: set[str] = set()
    for hit in fused["umls"]:
        cui = hit["cui"]
        if cui in seen:             # 去重
            continue
        seen.add(cui)

        triples.append(
            Triple(
                head=patient_id,    # 只留純 ID，前綴交由 head_type 記錄
                head_type="Patient",
                relation="HAS_DISEASE",
                tail=cui,           # 只留 CUI，本身就是疾病代碼
                tail_type="Disease",
                timestamp=None,
                source="UMLS",
            )
        )
        
    # 2) RAG‐generated PubMed triples
    abstracts = [h['abstract'] for h in fused['pubmed']]
    rag_json = generate_triples_local_llama2(
        patient_context=patient_context,
        abstracts=abstracts,
        umls_facts=fused['umls_facts'],
    )
    for d in rag_json:
        triples.append(Triple(**d))

    return triples

