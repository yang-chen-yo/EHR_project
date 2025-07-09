# === retrieval/pubmed_client.py ===
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional

import requests

__all__ = [
    "PubMedClient",
    "search_pubmed",
]


class PubMedClient:
    """簡易封裝 NCBI E‑utilities (PubMed)。"""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, email: str, api_key: Optional[str] = None):
        if not email:
            raise ValueError("`email` 為必填，以符合 NCBI E‑utilities 使用規範")
        self.email = email
        self.api_key = api_key

    # ---------------------------------------------------------------------
    # ESearch：依關鍵字取得 PMID 清單 (JSON 支援)
    # ---------------------------------------------------------------------
    def search(
        self,
        term: str,
        *,
        retmax: int = 5,
        sleep_sec: float = 0.34,
    ) -> List[str]:
        params = {
            "db": "pubmed",
            "term": term,
            "retmax": retmax,
            "retmode": "json",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        resp = requests.get(f"{self.BASE_URL}/esearch.fcgi", params=params, timeout=20)
        resp.raise_for_status()
        pmids = resp.json().get("esearchresult", {}).get("idlist", [])
        time.sleep(sleep_sec)
        return pmids

    # ---------------------------------------------------------------------
    # EFetch：依 PMID 抓取 XML，解析標題與摘要
    # ---------------------------------------------------------------------
    def fetch_abstracts(
        self,
        pmids: List[str],
        *,
        sleep_sec: float = 0.34,
    ) -> List[Dict[str, str]]:
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        resp = requests.get(f"{self.BASE_URL}/efetch.fcgi", params=params, timeout=30)
        resp.raise_for_status()
        xml_text = resp.text

        # 解析 XML
        root = ET.fromstring(xml_text)
        articles: List[Dict[str, str]] = []
        for art in root.findall(".//PubmedArticle"):
            pmid = art.findtext(".//PMID") or ""
            title = (art.findtext(".//ArticleTitle") or "").strip()
            # 摘要可能分多段 <AbstractText>
            abstract_parts = [seg.text or "" for seg in art.findall(".//AbstractText")]
            abstract = ' '.join(part.strip() for part in abstract_parts if part.strip())
            # 解析出版年份
            year = art.findtext(".//PubDate/Year") or None
            if not year:
                # fallback: 從 PMID 前 4 位嘗試
                year = pmid[:4] if pmid[:4].isdigit() else None
            articles.append({"pmid": pmid, "title": title, "abstract": abstract, "year": int(year) if year and year.isdigit() else None})

        time.sleep(sleep_sec)
        return articles


# -------------------------------------------------------------------------
# Convenience wrapper：多關鍵字批次搜尋
# -------------------------------------------------------------------------

def search_pubmed(
    keywords: List[str],
    *,
    email: str,
    api_key: Optional[str] = None,
    retmax: int = 5,
) -> List[Dict[str, str]]:
    client = PubMedClient(email=email, api_key=api_key)
    out: List[Dict[str, str]] = []
    for kw in keywords:
        pmids = client.search(kw, retmax=retmax)
        for art in client.fetch_abstracts(pmids):
            art["keyword"] = kw
            out.append(art)
    return out
