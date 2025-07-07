import os, sys
import unittest
from unittest.mock import patch, MagicMock

# 設定專案根目錄
test_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(test_dir, '..'))
sys.path.insert(0, root_dir)

from retrieval.pubmed_client import PubMedClient, search_pubmed

class TestPubMedClient(unittest.TestCase):
    def setUp(self):
        self.email = 'test@example.com'
        self.api_key = 'DUMMYKEY'
        self.client = PubMedClient(email=self.email, api_key=self.api_key)

    @patch('retrieval.pubmed_client.requests.get')
    def test_search(self, mock_get):
        # 模擬 esearch 回傳值
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            'esearchresult': {'idlist': ['12345', '67890']}
        }
        mock_get.return_value = mock_resp

        ids = self.client.search('diabetes', retmax=2)
        self.assertEqual(ids, ['12345', '67890'])
        mock_get.assert_called()

    @patch('retrieval.pubmed_client.requests.get')
    def test_fetch_abstracts(self, mock_get):
        # 模擬 efetch 回傳值
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            'result': {
                'uids': ['12345'],
                '12345': {'title': 'Test Title', 'abstract': 'Test Abstract'}
            }
        }
        mock_get.return_value = mock_resp

        articles = self.client.fetch_abstracts(['12345'])
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]['pmid'], '12345')
        self.assertEqual(articles[0]['title'], 'Test Title')
        self.assertEqual(articles[0]['abstract'], 'Test Abstract')

    @patch('retrieval.pubmed_client.PubMedClient.search')
    @patch('retrieval.pubmed_client.PubMedClient.fetch_abstracts')
    def test_search_pubmed_wrapper(self, mock_fetch, mock_search):
        mock_search.return_value = ['111']
        mock_fetch.return_value = [{'pmid': '111', 'title': 'T', 'abstract': 'A'}]

        res = search_pubmed(['kw1'], email=self.email, api_key=self.api_key)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]['keyword'], 'kw1')

if __name__ == '__main__':
    unittest.main()
