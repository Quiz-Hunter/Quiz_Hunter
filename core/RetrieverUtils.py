import core.BmHnsw as bh

def bm25_hnsw_retriever():
    retriever = bh.BM25HNSWRetriever(
        es_url='https://user:password@localhost:9200',
        ca_cert_path='http_ca.crt'
    )
    retriever.create_index_and_ingest("path/to/your/data.csv")
    results = retriever.search(query_text="請幫我找相關句子", top_k=5)
    return "bm25_retriever" # TODO

def vector_embedding_retriever():
    return "vector_retriever" # TODO

def hybrid_retriever():
    return "hybrid_retriever" # TODO 顆粒度不同
