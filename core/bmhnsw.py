import os
import json
import numpy as np
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


class BM25HNSWRetriever:
    def __init__(self, data_path, model_name="shibing624/text2vec-base-chinese"):
        self.data_path = data_path
        self.model = SentenceTransformer(model_name)
        self.data = []
        self.contents = []
        self.embeddings = None
        self.bm25 = None
        self.faiss_index = None

    def load_and_prepare(self):
        print(f"Loading JSON data from: {self.data_path}")
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        def build_content(q):
            parts = []
            if q.get("group_id"):
                parts.append(q.get("group_context", ""))
            parts.append(q["stem"])
            for k, v in q.get("options", {}).items():
                parts.append(f"({k}) {v}")
            return " ".join(parts)

        self.contents = [build_content(q) for q in self.data]

        print("Encoding embeddings with SentenceTransformer...")
        self.embeddings = self.model.encode(
            self.contents,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        print("Building BM25 index...")
        tokenized_corpus = [text.split(" ") for text in self.contents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print("Building FAISS HNSW index (Cosine similarity)...")
        dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        self.faiss_index.hnsw.efConstruction = 100
        self.faiss_index.add(self.embeddings)

    def search(self, query, top_k=5, alpha=0.5):
        if self.faiss_index is None or self.bm25 is None:
            raise RuntimeError("Please run load_and_prepare() first.")

        query_embedding = self.model.encode([query], normalize_embeddings=True)
        faiss_scores, faiss_ids = self.faiss_index.search(query_embedding, top_k * 10)
        faiss_scores, faiss_ids = faiss_scores[0], faiss_ids[0]

        bm25_scores = np.array(self.bm25.get_scores(query.split(" ")))
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)

        hybrid = []
        for idx, faiss_score in zip(faiss_ids, faiss_scores):
            bm25_score = bm25_scores[idx]
            score = alpha * faiss_score + (1 - alpha) * bm25_score
            hybrid.append((idx, score))

        hybrid = sorted(hybrid, key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in hybrid:
            q = self.data[idx]
            results.append({
                "id": q["id"],
                "year": q.get("year", "unknown"),
                "subject": q.get("subject", "unknown"),
                "content": self.contents[idx],
                "score": score
            })
        return results


if __name__ == "__main__":
    retriever = BM25HNSWRetriever("C:\\Users\\0524e\\OneDrive\\文件\\GitHub\\Quiz_Hunter\\Quiz_json\\all.json")  # ← JSON 題庫
    retriever.load_and_prepare()

    query = "下列哪一種物質與適當的催化劑共熱，可得到氧氣？"
    results = retriever.search(query, top_k=3, alpha=0.5)

    for r in results:
        print(f"{r['id']} | {r['year']}年 {r['subject']} | {r['score']:.2f} 分")
        print(f"→ {r['content']}\n")
