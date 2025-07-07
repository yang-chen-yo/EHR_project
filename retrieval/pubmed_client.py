import os
import time
import requests
from typing import List, Dict
from config import DATA_BASE_PATH

class PubMedClient:
    """
    使用 NCBI E-utilities 透過 API 檢索 PubMed 文獻摘要
    - 預設速率限制: 約3次/秒，可透過 sleep_sec 參數調整
    """
    BASE_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'

    def __init__(self, email: str):
        self.email = email

    def search(self, term: str, retmax: int = 5, sleep_sec: float = 0.34) -> List[str]:
        """
        依關鍵字搜尋並回傳 PMID 列表
        Args:
            term: 搜尋字串
            retmax: 最多回傳筆數
            sleep_sec: 呼叫後等待秒數，避免速率過高
        Returns:
            List of PMID strings
        """
        params = {
            'db': 'pubmed',
            'term': term,
            'retmax': retmax,
            'retmode': 'json',
            'email': self.email
        }
        resp = requests.get(f"{self.BASE_URL}/esearch.fcgi", params=params)
        resp.raise_for_status()
        data = resp.json()
        pmids = data.get('esearchresult', {}).get('idlist', [])
        time.sleep(sleep_sec)
        return pmids

    def fetch_abstracts(self, pmids: List[str], sleep_sec: float = 0.34) -> List[Dict[str, str]]:
        """
        依 PMID 列表抓取摘要與標題
        Args:
            pmids: PMID 列表
            sleep_sec: 呼叫後等待秒數
        Returns:
            List of dicts with keys: pmid, title, abstract
        """
        if not pmids:
            return []
        ids_str = ','.join(pmids)
        params = {
            'db': 'pubmed',
            'id': ids_str,
            'retmode': 'json',
            'email': self.email
        }
        resp = requests.get(f"{self.BASE_URL}/efetch.fcgi", params=params)
        resp.raise_for_status()
        data = resp.json()
        uids = data.get('result', {}).get('uids', [])
        articles: List[Dict[str, str]] = []
        for uid in uids:
            rec = data['result'].get(uid, {})
            title = rec.get('title', '').strip()
            abstract = rec.get('abstract', '')
            if isinstance(abstract, list):
                abstract = ' '.join(abstract)
            articles.append({'pmid': uid, 'title': title, 'abstract': abstract})
        time.sleep(sleep_sec)
        return articles


def search_pubmed(keywords: List[str], email: str) -> List[Dict[str, str]]:
    """
    Wrapper: 給定多個 keyword，用 PubMedClient 搜尋並回傳摘要
    """
    client = PubMedClient(email=email)
    results: List[Dict[str, str]] = []
    for kw in keywords:
        pmids = client.search(kw)
        articles = client.fetch_abstracts(pmids)
        for art in articles:
            art['keyword'] = kw
            results.append(art)
    return results
