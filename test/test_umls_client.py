# === tests/test_umls_client.py ===
import os, sys
import unittest

# 設定專案根目錄並切換 cwd，確保相對路徑正確
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)
os.chdir(root_dir)

from retrieval.umls_client import UMLSClient, query_umls

class TestUMLSClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 假設 tests/test_umls_data 底下含有 umls.csv, relation.txt, concept.txt, concept_name.txt
        test_data_dir = os.path.join(root_dir, 'tests', 'test_umls_data')
        cls.client = UMLSClient(umls_dir=test_data_dir)

    def test_relations_loaded(self):
        self.assertIn('may_cause', self.client.relations)
        self.assertIsInstance(self.client.relations, list)

    def test_concepts_and_names(self):
        self.assertIn('C0000039', self.client.concepts)
        self.assertEqual(
            self.client.concept_names.get('C0000039'),
            '1,2-dipalmitoylphosphatidylcholine'
        )

    def test_code_to_cuis_and_query(self):
        # 假設 test_umls_data/umls.csv 包含 code 'A01' -> 'C0000039'
        mapping = self.client.code2cuis
        self.assertIn('A01', mapping)
        results = self.client.query_by_codes(['A01'])
        self.assertTrue(any(r['cui'] == 'C0000039' for r in results))

    def test_query_umls_wrapper(self):
        out = query_umls(['A01'], umls_dir=os.path.join(root_dir, 'tests', 'test_umls_data'))
        self.assertIsInstance(out, list)
        self.assertEqual(out[0]['name'], self.client.concept_names[out[0]['cui']])

if __name__ == '__main__':
    unittest.main()


# === retrieval/pubmed_client.py ===
import os
import requests
from typing import List, Dict
from config import DATA_BASE_PATH

class PubMedClient:
    """
    使用 NCBI E-utilities API 檢索 PubMed 文獻摘要
    """
    BASE_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'

    def __init__(self, email: str, api_key: str = None):
        self.email = email
        self.api_key = api_key

    def search(self, term: str, retmax: int = 5) -> List[str]:
        params = {
            'db': 'pubmed',
            'term': term,
            'retmax': retmax,
            'retmode': 'json',
            'email': self.email
        }
        if self.api_key:
            params['api_key'] = self.api_key
        resp = requests.get(f"{self.BASE_URL}/esearch.fcgi", params=params)
        resp.raise_for_status()
        data = resp.json()
        return data['esearchresult'].get('idlist', [])

    def fetch_abstracts(self, pmids: List[str]) -> List[Dict[str, str]]:
        ids = ','.join(pmids)
        params = {
            'db': 'pubmed',
            'id': ids,
            'retmode': 'json',
            'email': self.email
        }
        if self.api_key:
            params['api_key'] = self.api_key
        resp = requests.get(f"{self.BASE_URL}/efetch.fcgi", params=params)
        resp.raise_for_status()
        data = resp.json()
        articles = []
        for uid in data.get('result', {}).get('uids', []):
            rec = data['result'].get(uid, {})
            articles.append({
                'pmid': uid,
                'title': rec.get('title', ''),
                'abstract': rec.get('abstract', '')
            })
        return articles


def search_pubmed(keywords: List[str], email: str, api_key: str = None) -> List[Dict[str, str]]:
    client = PubMedClient(email=email, api_key=api_key)
    results: List[Dict[str, str]] = []
    for kw in keywords:
        pmids = client.search(kw)
        abstracts = client.fetch_abstracts(pmids)
        for art in abstracts:
            art['keyword'] = kw
            results.append(art)
    return results
