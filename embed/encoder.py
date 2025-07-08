# === embed/encoder.py ===
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class Encoder:
    """
    封裝 Sentence-BERT 將文本轉成向量
    """
    def __init__(self, model_name: str = "pritamdeka/SapBERT-from-PubMedBERT-fulltext"):
        # 載入預訓練模型
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        texts: List of strings
        回傳 shape = (len(texts), dim)
        """
        # 正規化輸入
        clean_texts = [t if isinstance(t, str) else str(t) for t in texts]
        # 轉向量並正規化
        embeddings = self.model.encode(
            clean_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings