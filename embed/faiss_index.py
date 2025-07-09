# === embed/faiss_index.py ===
import faiss
import numpy as np

typedef = np.ndarray

class FaissIndex:
    """
    建立並查詢 FAISS 向量索引
    """
    def __init__(self, dim: int):
        # 使用內積 (normalized embeddings) 等同 cosine 相似度
        self.index = faiss.IndexFlatIP(dim)

    def build(self, vectors: np.ndarray):
        """
        vectors: numpy array shape [N, dim]
        """
        self.index.add(vectors)

    def search(self, query_vec: np.ndarray, k: int = 5):
        """
        query_vec: shape [dim] 或 [1, dim]
        回傳 (indices, scores)
        """
        q = query_vec.reshape(1, -1).astype('float32')
        D, I = self.index.search(q, k)
        # I: [1,k], D: [1,k]
        return I[0].tolist(), D[0].tolist()