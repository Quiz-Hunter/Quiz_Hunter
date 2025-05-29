import numpy as np
from sentence_transformers import SentenceTransformer, util

class SimilaritySearcher:
    def __init__(self, npz_paths, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)
        self.embs = []
        # 用一個 dict 一次讀完所有欄位
        self.data = {k: [] for k in (
            "embed_texts","stem_texts","contexts","options","years","subjects")}
        for path in npz_paths:
            arr = np.load(path, allow_pickle=True)
            self.embs.append(arr["embs"])
            for k in self.data:
                self.data[k].extend(arr[k])
        self.embs = np.vstack(self.embs)

    def search(self, context, stem, top_k=5):
        # 用題組上下文+題幹做 query
        query = f"{context} {stem}".strip() if context else stem.strip()
        q_emb  = self.model.encode(query, convert_to_numpy=True)
        hits   = util.semantic_search(q_emb, self.embs, top_k=top_k*2)[0]

        print("\n🚀 相似題目結果（題組+題幹+選項對比）：\n")
        shown = 0
        for hit in hits:
            idx, score = hit["corpus_id"], hit["score"]
            # 跳過和 query 一模一樣的
            if self.data["embed_texts"][idx].strip() == query.strip():
                continue

            shown += 1
            print(f"{shown}. 📌 年度：{self.data['years'][idx]} | 科目：{self.data['subjects'][idx]} | 相似度：{score:.4f}")
            if self.data["contexts"][idx]:
                print(f"    題組背景：{self.data['contexts'][idx]}")
            print(f"    題幹：{self.data['stem_texts'][idx]}")
            print("    選項：")
            for lab, txt in self.data["options"][idx].items():
                print(f"      ({lab}) {txt}")
            print()
            if shown >= top_k:
                break

        if shown == 0:
            print("⚠️ 沒有找到不同於輸入的相似題目。")