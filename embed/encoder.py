# === embed/encoder.py ===
from typing import List
import numpy as np
import config
from google import genai
from google.genai import types

# 初始化 Gemini Client，帶入 config 裡的 API key
client = genai.Client(api_key=config.GOOGLE_API_KEY)

class Encoder:
    """
    使用 Google Gemini API 的嵌入模型來轉換文字成向量
    """
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBED_MODEL_NAME
        self.batch_size = config.EMBED_BATCH_SIZE
        self.task_type  = config.EMBED_TASK_TYPE

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        texts: List of strings
        回傳 shape = (len(texts), dim)
        """
        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            for text in batch:
                resp = client.models.embed_content(
                    model=self.model_name,
                    contents=[text],
                    config=types.EmbedContentConfig(
                        task_type=self.task_type
                    )
                )
                # resp.embeddings 是 List[ContentEmbedding]
                emb_obj = resp.embeddings[0]
                embeddings.append(emb_obj.values)

        # 轉成 numpy 陣列並指定 float32，供 FAISS 使用
        return np.array(embeddings, dtype=np.float32)

