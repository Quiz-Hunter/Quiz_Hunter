import core.BmHnsw as bh
import glob
from core.SimilaritySearcher import SimilaritySearcher

def bm25_hnsw_retriever():
    retriever = bh.BM25HNSWRetriever("C:\\Users\\0524e\\OneDrive\\文件\\GitHub\\Quiz_Hunter\\Quiz_json\\all.json")  # ← JSON 題庫
    retriever.load_and_prepare()
    return retriever


def vector_embedding_retriever():
    npz_files = glob.glob("./Quiz_clean_Embedding_npz/*.npz")
    if not npz_files:
        print(" 無法找到任何 .npz 向量檔案，請確認後再試一次。")
        return

    # 載入向量檔
    searcher = SimilaritySearcher(npz_files)
    return "vector_retriever" # TODO


