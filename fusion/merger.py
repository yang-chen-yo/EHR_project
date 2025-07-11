from typing import List, Dict
from datetime import datetime
import os, json

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
    patient_fields: Dict[str, list],
    k_umls: int = K_UMLS,
    k_pubmed: int = K_PUBMED,
) -> Dict[str, List]:
    """
    向量化 UMLS + PubMed 檢索，並計算各自邊權重。

    Returns:
      {
        'umls':   List[{'cui','name','score'}],  # score 為 cosine 相似度
        'pubmed': List[{'pmid','title','abstract','score','year'}],  # score 為 sim×recency
      }
    """
    encoder = Encoder()
    qvec = encoder.encode([patient_text])[0]

    # UMLS
    client_u = UMLSClient(umls_dir)
    cuis  = client_u.concepts
    names = [client_u.concept_names[c] for c in cuis]
    name_vecs = encoder.encode(names)
    idx_u = FaissIndex(name_vecs.shape[1])
    idx_u.build(name_vecs)
    ids_u, sims_u = idx_u.search(qvec, k_umls)
    umls_hits = [
        {'cui':cuis[i], 'name':names[i], 'score':float(sims_u[j])}
        for j,i in enumerate(ids_u)
    ]

    # PubMed: 根據患者字段逐 concept 檢索
    client_p = PubMedClient(email=pubmed_email)
    topics = []
    for field in ('conditions','procedures','drugs'):
        for sub in patient_fields.get(field,[]):
            topics += [c['name'] for c in sub]

    all_pmids = []
    for name in topics:
        all_pmids += client_p.search(name, retmax=k_pubmed)
    all_pmids = list(dict.fromkeys(all_pmids))[: k_pubmed*max(len(topics),1)]
    arts = client_p.fetch_abstracts(all_pmids)

    pubmed_hits = []
    for art in arts:
        vec = encoder.encode([art['abstract']])[0]
        sim = cosine_similarity(qvec, vec)
        score = score_pubmed_hit(sim, art.get('year') or datetime.now().year)
        pubmed_hits.append({**art, 'score':score})
    pubmed_hits.sort(key=lambda x: x['score'], reverse=True)

    return {'umls':umls_hits,'pubmed':pubmed_hits}


def merge_to_triples(
    patient_id: str,
    fused: Dict[str, List],
    patient_context: str,
    patient_fields: Dict[str, list],
    output_dir: str = 'triples_output',
    accumulate: bool = True,
) -> List[Triple]:
    """
    合併 UMLS 與 RAG(PubMed) 三元組，並將邊權重寫入 Triple.weight。

    若 accumulate=True，則回傳包含所有 Triple 的 list；
    否則僅寫檔不回傳。
    """
    os.makedirs(os.path.join(output_dir, patient_id), exist_ok=True)
    triples: List[Triple] = []

    # UMLS 三元組 + weight
    for hit in fused['umls']:
        triples.append(
            Triple(
                head=patient_id, head_type='Patient',
                relation='HAS_DISEASE', tail=hit['cui'], tail_type='Disease',
                source='UMLS', weight=hit['score']
            )
        )

    # RAG 三元組 + weight (取首篇摘要最高 score)
    best_score = fused['pubmed'][0]['score'] if fused['pubmed'] else None
    abstracts = [h['abstract'] for h in fused['pubmed']]
    rag_list = generate_triples_local_llama2(
        patient_context=patient_context,
        abstracts=abstracts,
        umls_facts=None
    )
    for item in rag_list:
        triples.append(
            Triple(
                head=item['head'], head_type=item['head_type'],
                relation=item['relation'], tail=item['tail'], tail_type=item['tail_type'],
                timestamp=item.get('timestamp'), source='PubMed', weight=best_score
            )
        )
        if accumulate:
            # 可選：將每個 concept 的 RAG 輸出另存檔案
            pdir = os.path.join(output_dir, patient_id)
            fname = f"{item['relation']}_{item['tail']}.json"
            with open(os.path.join(pdir, fname),'w',encoding='utf-8') as f:
                json.dump(item,f,ensure_ascii=False,indent=2)

    return triples if accumulate else []