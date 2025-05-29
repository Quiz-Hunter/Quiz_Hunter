import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


class BM25HNSWRetriever:
    def __init__(self, data_path, model_name="distiluse-base-multilingual-cased-v1"):
        self.data_path = data_path
        self.model = SentenceTransformer(model_name)
        self.data = None
        self.embeddings = None
        self.bm25 = None
        self.faiss_index = None

    def load_and_prepare(self):
        df = pd.read_json(self.data_path, lines=True)  # 改用 JSONL 格式，或讀 json

        if not {'id', 'stem', 'options'}.issubset(df.columns):
            raise ValueError("JSON 檔案必須包含 'id', 'stem', 'options' 欄位")

        # 合併題幹與選項為一個 content 欄位
        def build_content(row):
            opts = row["options"]
            opts_str = " ".join([f"{key}:{val}" for key, val in opts.items()])
            return f"{row['stem']} 選項：{opts_str}"

        df["content"] = df.apply(build_content, axis=1)
        df["date"] = "unknown"  # 沒有日期就補上 placeholder

        self.data = df

        print("Encoding embeddings with SentenceTransformer...")
        self.embeddings = self.model.encode(
            df['content'].tolist(),
            show_progress_bar=True,
            normalize_embeddings=True
        )

        print("Building BM25 index...")
        tokenized_corpus = [doc.split(" ") for doc in df['content']]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print("Building FAISS HNSW (Cosine) index...")
        dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        self.faiss_index.hnsw.efConstruction = 100
        self.faiss_index.add(self.embeddings)


    def search(self, query, top_k=5, alpha=0.5):
        """
        alpha: 0~1, 控制向量分數與 BM25 的混合比例
        """
        if self.faiss_index is None or self.bm25 is None:
            raise RuntimeError("Index not built yet. Please call `load_and_prepare()` first.")

        # 向量查詢 (cosine similarity 透過內積)
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        faiss_scores, faiss_ids = self.faiss_index.search(query_embedding, top_k * 10)
        faiss_scores = faiss_scores[0]
        faiss_ids = faiss_ids[0]

        # BM25 查詢 + normalization
        bm25_scores = np.array(self.bm25.get_scores(query.split(" ")))
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)

        # 混合分數計算
        hybrid_results = []
        for idx, faiss_score in zip(faiss_ids, faiss_scores):
            bm25_score = bm25_scores[idx]
            hybrid_score = alpha * faiss_score + (1 - alpha) * bm25_score
            hybrid_results.append((idx, hybrid_score))

        # 排序與輸出
        hybrid_results = sorted(hybrid_results, key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in hybrid_results:
            row = self.data.iloc[idx]
            results.append({
                "id": row["id"],
                "content": row["content"],
                "date": row["date"],
                "score": score
            })

        return results


if __name__ == '__main__':
    retriever = BM25HNSWRetriever(data_path="test.csv")
    retriever.load_and_prepare()

    query = "銅線與硝酸反應會產生氣體與硝酸銅，請問需要多少硝酸體積？"
    results = retriever.search(query, top_k=3, alpha=0.5)

    for r in results:
        print(f"{r['id']} | {r['date']} | {r['score']:.2f} | {r['content']}")
