import json, numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    def __init__(self, json_path, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.json_path = json_path
        self.model = SentenceTransformer(model_name)


    def make_embedding_text(self, q):
        # 僅用題組上下文 + 題幹 + 選項
        parts = []
        if q.get("group_id"):
            parts.append(q["group_context"])
        parts.append(q["stem"])
        # 把所有選項也加進向量化文字裡
        for lab, txt in q.get("options", {}).items():
            parts.append(f"({lab}) {txt}")
        return " ".join(parts).strip()
    
    
    def generate_embeddings(self, output_npz_path):
        with open(self.json_path, encoding="utf-8") as f:
            questions = json.load(f)

        ids         = [q["id"]                for q in questions]
        embed_texts = [self.make_embedding_text(q) for q in questions]
        stem_texts  = [q["stem"]              for q in questions]
        contexts    = [q.get("group_context","") for q in questions]
        options     = [q.get("options",{})     for q in questions]
        years       = [q.get("year","")        for q in questions]
        subjects    = [q.get("subject","")     for q in questions]

        embs = self.model.encode(embed_texts, convert_to_numpy=True, batch_size=32)

        np.savez(
            output_npz_path,
            ids=ids,
            embs=embs,
            embed_texts=embed_texts,
            stem_texts=stem_texts,
            contexts=contexts,
            options=options,
            years=years,
            subjects=subjects
        )
        print(f"已存 embeddings（題組+題幹+選項，含年度、科目）至 {output_npz_path}")

