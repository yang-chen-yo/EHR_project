import os
import sys
import unittest

# 把專案根目錄加入到模組搜尋路徑
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from fusion.triple_generation_hf import generate_triples_local_llama2


class TestRealRAGIntegration(unittest.TestCase):
    def test_real_llm_default_model(self):
        """
        使用預設 Llama-2 RAG 模型實際產生 triples，並把結果列印出來。
        若模型載入或推理失敗，印出錯誤並讓測試失敗。
        """
        patient_context = (
            "PatientID: P0001; Visit Date: 2025-07-09; "
            "Diagnoses: I10; Medications: 2025-07-10 Lisinopril (C09AA02); "
            "Labs: BloodPressure=160/100 mmHg."
        )
        abstracts = [
            "Lisinopril has been shown to effectively reduce systolic blood pressure in patients with hypertension (I10)."
        ]

        try:
            triples = generate_triples_local_llama2(
                patient_context=patient_context,
                abstracts=abstracts,
                umls_facts=None
            )
        except Exception as e:
            # 直接印出錯誤並讓測試失敗，便於除錯
            print(f"\n[ERROR] LLM 推理失敗：{e}\n")
            self.fail("LLM 推理失敗，請確認模型或 GPU 環境")
            return  # 保險起見（理論上不會執行到）

        # 印出 LLM 的實際輸出
        print("\n=== Generated Triples ===")
        for t in triples:
            print(t)
        print("=========================\n")

        # 基本格式檢查
        self.assertIsInstance(triples, list)
        self.assertGreater(len(triples), 0, "Should generate at least one triple")
        first = triples[0]
        self.assertIsInstance(first, dict)
        for key in ["head", "head_type", "relation", "tail", "tail_type", "source"]:
            self.assertIn(key, first)


if __name__ == '__main__':
    unittest.main()

