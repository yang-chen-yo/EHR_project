# === retrieval/pubmed_client.py  (帶自動重試與指數退避) ===
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

__all__ = ["PubMedClient", "search_pubmed"]


class PubMedClient:
    """簡易封裝 NCBI E-utilities (PubMed)，內建自動重試 + 指數退避。"""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(
        self,
        email: str,
        api_key: Optional[str] = None,
        *,
        max_retries: int = 5,
        backoff_factor: float = 1.0,
    ):
        """
        Args
        ----
        email          : 你的聯絡信箱（遵守 NCBI 規範）
        api_key        : NCBI 個人 API Key（可提高限額，選填）
        max_retries    : 連線或 5xx 失敗時，最高重試次數
        backoff_factor : 指數退避係數；1→1,2,4,8… 秒
        """
        if not email:
            raise ValueError("`email` 為必填，以符合 NCBI E-utilities 使用規範")
        self.email = email
        self.api_key = api_key

        # -------- 建立帶重試的 Session ----------
        retry_cfg = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry_cfg)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    # ------------------------------------------------------------------
    # ESearch：依關鍵字取得 PMID 清單
    # ------------------------------------------------------------------
    def search(
        self,
        term: str,
        *,
        retmax: int = 5,
        sleep_sec: float = 0.11,
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

        resp = self.session.get(
            f"{self.BASE_URL}/esearch.fcgi", params=params, timeout=20
        )
        resp.raise_for_status()
        pmids = resp.json().get("esearchresult", {}).get("idlist", [])
        time.sleep(sleep_sec)  # 遵守 NCBI ≤3 req/s
        return pmids

    # ------------------------------------------------------------------
    # EFetch：依 PMID 抓取 XML，解析標題與摘要
    # ------------------------------------------------------------------
    def fetch_abstracts(
        self,
        pmids: List[str],
        *,
        sleep_sec: float = 0.11,
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

        resp = self.session.get(
            f"{self.BASE_URL}/efetch.fcgi", params=params, timeout=30
        )
        resp.raise_for_status()
        xml_text = resp.text

        root = ET.fromstring(xml_text)
        articles: List[Dict[str, str]] = []
        for art in root.findall(".//PubmedArticle"):
            pmid = art.findtext(".//PMID") or ""
            title = (art.findtext(".//ArticleTitle") or "").strip()

            # 摘要可能分多段
            abstract_parts = [seg.text or "" for seg in art.findall(".//AbstractText")]
            abstract = " ".join(p.strip() for p in abstract_parts if p.strip())

            # 先嘗試取 ArticleDate，若無再退回 PubDate
            ad = art.find(".//ArticleDate")
            if ad is not None:
                ay = ad.findtext("Year")
                am = ad.findtext("Month")
                ady = ad.findtext("Day")
            else:
                pub = art.find(".//PubDate")
                ay = pub.findtext("Year") or None
                am = pub.findtext("Month") or None
                ady = pub.findtext("Day") or None

            # 將英文月份縮寫轉成數字
            month_map = {
                'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06',
                'Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'
            }
            if ay and am and ady:
                m = month_map.get(am, am.zfill(2))
                d = ady.zfill(2)
                timestamp = f"{ay}-{m}-{d}"
            elif ay:
                timestamp = f"{ay}-01-01"
            else:
                timestamp = None

            articles.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "year": int(ay) if ay and ay.isdigit() else None,
                    "timestamp": timestamp,
                }
            )

        time.sleep(sleep_sec)
        return articles


# ----------------------------------------------------------------------
# Convenience wrapper：多關鍵字批次搜尋
# ----------------------------------------------------------------------
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

