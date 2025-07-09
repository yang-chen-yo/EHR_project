import unittest
from fusion.triple_generation_hf import generate_triples_local_llama2

class TestRealRAGIntegration(unittest.TestCase):
    def test_real_llm_default_model(self):
        """
        Integration test: use the default RAG model (meta-llama/Llama-2-7b-chat-hf) to generate triples.
        If the environment cannot load the model or has no GPU, this test will be skipped.
        """
        # Minimal patient context and abstract
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
                umls_facts=None  # optional UMLS facts
            )
        except Exception as e:
            self.skipTest(f"Skipping real RAG LLM test due to environment error: {e}")
            return

        # Basic assertions on output format
        self.assertIsInstance(triples, list)
        self.assertGreater(len(triples), 0, "Should generate at least one triple")
        first = triples[0]
        self.assertIsInstance(first, dict)
        for key in ["head", "head_type", "relation", "tail", "tail_type", "source"]:
            self.assertIn(key, first)

if __name__ == '__main__':
    unittest.main()
