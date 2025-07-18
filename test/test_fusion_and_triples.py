import os
import sys
import tempfile
import pickle
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

# ========== 調整路徑 ==========
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

import numpy as np

from fusion.scoring import cosine_similarity, recency_weight, score_pubmed_hit
from fusion.merger import fuse_and_score, merge_to_triples
from fusion.triple_generation_hf import generate_triples_local_llama2
from kg.triple import (
    samples_to_triples,
    load_preprocessed_samples,
    load_triples_from_samples,
    Triple,
)
from config import sample_dataset_path


# -------------------------------------------------------------------
class TestFusionScoring(unittest.TestCase):
    def test_cosine_similarity(self):
        a = np.array([1, 0, 0])
        b = np.array([1, 0, 0])
        sim = cosine_similarity(a, b)
        print("\n[cosine_similarity] =", sim)
        self.assertAlmostEqual(sim, 1.0)

    def test_recency_weight(self):
        current = datetime.now().year
        w = recency_weight(current)
        print("[recency_weight] =", w)
        self.assertAlmostEqual(w, 1.0)

    def test_score_pubmed_hit(self):
        current = datetime.now().year
        score = score_pubmed_hit(1.0, current)
        print("[score_pubmed_hit] =", score)
        self.assertAlmostEqual(score, 1.0)


# -------------------------------------------------------------------
class TestKGTriples(unittest.TestCase):
    def test_samples_to_triples(self):
        samples = [
            {
                'patient_id': 'p1',
                'conditions': ['C1'],
                'procedures': ['P1'],
                'drugs': ['D1'],
                'visit_time': '2025-07-08'
            }
        ]
        triples = samples_to_triples(samples)
        print("\n[samples_to_triples]")
        for t in triples:
            print(" ", t)
        self.assertEqual(len(triples), 3)
        self.assertEqual(triples[0].head, 'Patient:p1')
        self.assertEqual(triples[0].relation, 'HAS_DISEASE')

    def test_load_triples_from_samples(self):
        # 建立暫存 pickle
        samples = [
            {'patient_id': 'p2', 'conditions': ['C2'], 'procedures': [], 'drugs': [], 'visit_time': None}
        ]
        tmp = tempfile.NamedTemporaryFile(delete=False)
        pickle.dump(samples, tmp)
        tmp.close()

        with patch('config.sample_dataset_path', return_value=tmp.name):
            triples = load_triples_from_samples('x', 'y')
        os.unlink(tmp.name)

        print("\n[load_triples_from_samples]")
        for t in triples:
            print(" ", t)
        self.assertEqual(len(triples), 1)
        self.assertEqual(triples[0].tail, 'Disease:C2')


# -------------------------------------------------------------------
class TestTripleGenerationHF(unittest.TestCase):
    @patch('fusion.triple_generation_hf.pipeline')
    @patch('fusion.triple_generation_hf.AutoModelForCausalLM')
    @patch('fusion.triple_generation_hf.AutoTokenizer')
    def test_generate_triples_local_llama2(self, mock_tok, mock_model, mock_pipe):
        fake_out = (
            'xxx[{{"head":"Patient:p","head_type":"Patient","relation":"R",'
            '"tail":"T","tail_type":"Type","timestamp":"2025-07-01","source":"X"}}]'
        )
        fake_gen = MagicMock(return_value=[{'generated_text': fake_out}])
        mock_pipe.return_value = fake_gen

        triples = generate_triples_local_llama2('ctx', ['abs'], ['fact'])
        print("\n[generate_triples_local_llama2]", triples)

        self.assertIsInstance(triples, list)
        self.assertEqual(triples[0]['relation'], 'R')


# -------------------------------------------------------------------
class TestMerger(unittest.TestCase):
    @patch('fusion.merger.UMLSClient')
    @patch('fusion.merger.PubMedClient')
    @patch('fusion.merger.generate_triples_local_llama2')
    def test_fuse_and_merge(self, mock_gen, mock_pub, mock_umls):
        # 偽造 UMLSClient
        u = mock_umls.return_value
        u.concepts = ['C1']
        u.concept_names = {'C1': 'Name1'}
        u.relation_triples = []

        # 偽造 PubMedClient
        p = mock_pub.return_value
        p.search.return_value = ['123']
        p.fetch_abstracts.return_value = [{
            'pmid': '123',
            'title': 'T',
            'abstract': 'A',
            'year': datetime.now().year
        }]

        # RAG 生成結果
        mock_gen.return_value = [{
            'head': 'Patient:p1', 'head_type': 'Patient', 'relation': 'R',
            'tail': 'T', 'tail_type': 'Type', 'timestamp': None, 'source': 'PubMed'
        }]

        fused = fuse_and_score('ctx', 'dir', 'email')
        triples = merge_to_triples('p1', fused, 'ctx')

        print("\n[fuse_and_score] =>", fused)
        print("[merge_to_triples]")
        for t in triples:
            print(" ", t)

        self.assertEqual(len(triples), 2)
        self.assertEqual(triples[0].relation, 'HAS_DISEASE')
        self.assertEqual(triples[1].relation, 'R')


# -------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()

