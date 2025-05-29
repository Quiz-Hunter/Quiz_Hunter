import core.BmHnsw as bh

def bm25_hnsw_retriever():
    retriever = bh.BM25HNSWRetriever("C:\\Users\\0524e\\OneDrive\\文件\\GitHub\\Quiz_Hunter\\Quiz_json\\all.json")  # ← JSON 題庫
    return retriever


def vector_embedding_retriever():
    return "vector_retriever" # TODO


