import unittest
from data.loader import load_dataset
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
        # 檢查 samples 為非空列表
        self.assertIsInstance(self.samples, list)
        self.assertGreater(len(self.samples), 0)
        # 檢查第一筆 sample 結構
        first = self.samples[0]
        expected_keys = {"visit_id", "patient_id", "conditions", "procedures", "drugs", "label"}
        self.assertTrue(expected_keys.issubset(first.keys()))
        self.assertIsInstance(first['visit_id'], int)
        self.assertTrue(isinstance(first['patient_id'], (int, str)))
        self.assertIsInstance(first['conditions'], list)
        self.assertIsInstance(first['procedures'], list)
        self.assertIsInstance(first['drugs'], list)
        self.assertIsInstance(first['label'], int)

    def test_history_accumulation(self):
        # 檢查 conditions 與 procedures 隨時間累積
        for sample in self.samples:
            cond_hist = sample['conditions']
            proc_hist = sample['procedures']
            # 確保列表長度非遞減
            for i in range(len(cond_hist)-1):
                self.assertLessEqual(len(cond_hist[i]), len(cond_hist[i+1]))
            for i in range(len(proc_hist)-1):
                self.assertLessEqual(len(proc_hist[i]), len(proc_hist[i+1]))

    def test_label_values(self):
        # 檢查 label 僅為 0 或 1，且至少包含兩種值
        labels = [s['label'] for s in self.samples]
        for lbl in labels:
            self.assertIn(lbl, (0, 1))
        self.assertIn(0, labels)
        self.assertIn(1, labels)

if __name__ == '__main__':
    unittest.main()
