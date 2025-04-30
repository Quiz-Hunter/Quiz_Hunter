from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm
import os


class BM25HNSWRetriever:
    def __init__(self, es_url, ca_cert_path, index_name="bm25_hnsw_index"):
        self.index_name = index_name
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        self.es = Elasticsearch(es_url, ca_certs=ca_cert_path)

    def create_index_and_ingest(self, data_path):
        if self.es.indices.exists(index=self.index_name):
            print(f"Index '{self.index_name}' already exists.")
            return

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"CSV file not found: {data_path}")

        df = pd.read_csv(data_path)
        if not {'id', 'content', 'date'}.issubset(df.columns):
            raise ValueError("CSV must have 'id', 'content', and 'date' columns.")

        embeddings = self.model.encode(df['content'].tolist(), convert_to_tensor=True, normalize_embeddings=False, show_progress_bar=True)
        data = [
            {
                'id': row['id'],
                'content': row['content'],
                'date': row['date'],
                'embeddings': embedding.tolist()
            }
            for row, embedding in zip(df.to_dict('records'), embeddings)
        ]

        settings = {
            "settings": {
                "number_of_shards": 1
            },
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text"
                    },
                    "embeddings": {
                        "type": "dense_vector",
                        "dims": 512,
                        "index": True,
                        "similarity": "cosine",
                        "index_options": {
                            "type": "hnsw",
                            "m": 32,
                            "ef_construction": 100
                        }
                    },
                    "date": {
                        "type": "date",
                        "format": "yyyy-MM-dd HH:mm:ss.SSS"
                    }
                }
            }
        }

        self.es.indices.create(index=self.index_name, body=settings)

        for doc in tqdm(data, desc="Indexing data"):
            self.es.index(index=self.index_name, id=doc['id'], document=doc, refresh=False)
        self.es.indices.refresh(index=self.index_name)

    def search(self, query_text, top_k=5):
        query_vector = self.model.encode(query_text, convert_to_tensor=True, normalize_embeddings=False).tolist()

        es_query = {
            "knn": {
                "field": "embeddings",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": top_k * 10,
                "boost": 0.5
            },
            "query": {
                "match": {
                    "content": {
                        "query": query_text,
                        "boost": 0.5
                    }
                }
            },
            "size": top_k
        }

        response = self.es.search(index=self.index_name, body=es_query)
        return response['hits']['hits']
