import os, sys
# 設定專案根目錄並切換 cwd，確保相對路徑正確
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)
os.chdir(root_dir)

import unittest
from data.loader import load_mimic4_dataset as load_dataset
from data.preprocess import preprocess_samples

class TestPreprocess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 只載入少量資料以加快測試
        cls.dataset = load_dataset(load_processed=False, dataset="mimic4", task="mortality")
        # 限制前 5 位患者
        cls.sample_dataset = cls.dataset[:5]
        cls.samples = preprocess_samples(cls.sample_dataset)

    def test_samples_structure(self):
        # samples 為非空列表，且第一筆為 dict，具必要 keys
        self.assertIsInstance(self.samples, list)
        self.assertGreater(len(self.samples), 0)
        first = self.samples[0]
        self.assertIsInstance(first, dict)
        expected_keys = {"visit_id", "patient_id", "conditions", "procedures", "drugs", "label"}
        self.assertTrue(expected_keys.issubset(first.keys()))

    def test_samples_types(self):
        # 檢查各欄位型態
        first = self.samples[0]
        self.assertTrue(isinstance(first['visit_id'], (int, str)))
        self.assertTrue(isinstance(first['patient_id'], (int, str)))
        self.assertIsInstance(first['conditions'], list)
        self.assertIsInstance(first['procedures'], list)
        self.assertIsInstance(first['drugs'], list)
        self.assertTrue(isinstance(first['label'], int))

    def test_conditions_and_procedures_lists(self):
        # 確保 conditions 和 procedures 為 list-of-lists
        for sample in self.samples:
            cond_hist = sample['conditions']
            proc_hist = sample['procedures']
            self.assertTrue(isinstance(cond_hist, list))
            self.assertTrue(all(isinstance(v, (list, tuple)) for v in cond_hist))
            self.assertTrue(isinstance(proc_hist, list))
            self.assertTrue(all(isinstance(v, (list, tuple)) for v in proc_hist))

    def test_label_values(self):
        # label 僅為 0 或 1
        labels = [s['label'] for s in self.samples]
        for lbl in labels:
            self.assertIn(lbl, (0, 1))

if __name__ == '__main__':
    unittest.main()

