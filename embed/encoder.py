# === embed/encoder.py ===
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class Encoder:
    """
    封裝 Sentence-BERT （SapBERT）將文本轉成向量
    """
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 batch_size: int = 32):
        # 載入預訓練模型
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        將多條文本轉成向量（shape=(len(texts), dim)）。
        
        1) 轉成字串列表  
        2) 批次 encode  
        3) L2 正規化  
        """
        # 正規化輸入
        clean_texts = [t if isinstance(t, str) else str(t)
                       for t in texts]

        # 轉向量
        embeddings = self.model.encode(
            clean_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings

