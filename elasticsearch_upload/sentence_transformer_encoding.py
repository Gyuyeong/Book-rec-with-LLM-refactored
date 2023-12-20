from sentence_transformers import SentenceTransformer
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm


model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
input_filename = f"csv file directory to upload"
chunksize = 50
index_name = "data"

es = Elasticsearch(
    ["https://115.71.239.131:9200"],
    basic_auth=("elastic", "HWH1rJdFReoOA8i-NPiy"),
    verify_certs=False,
    timeout=30,
    max_retries=10,
    retry_on_timeout=True,
)

setting = {
    "analysis": {
        "analyzer": {
            "nori_analyzer": {
                "type": "nori",
                "tokenizer": "nori_mixed",
                # "filter": ["lowercase"]
            }
        },
        "tokenizer": {
            "nori_mixed": {
                "type": "nori_tokenizer",
                "decompound_mode": "mixed",
                "user_dictionary": "test_unique.txt",
            }
        },
    },
}

mapping = {
    "properties": {
        "author": {
            "type": "text",
            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            "analyzer": "nori_analyzer",
        },
        "category": {"type": "text", "analyzer": "nori_analyzer"},
        "introduction": {
            "type": "text",
            "analyzer": "nori_analyzer",
            "term_vector": "with_positions_offsets_payloads",
        },
        "publisher": {
            "type": "text",
            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            "analyzer": "nori_analyzer",
        },
        "title": {
            "type": "text",
            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            "analyzer": "nori_analyzer",
        },
        "publish_date": {
            "type": "date",
        },
        "isbn": {"type": "unsigned_long"},
        "toc": {"type": "text", "analyzer": "nori_analyzer"},
        "embedding": {
            "type": "dense_vector",
            "dims": 768,
            "index": True,
            "similarity": "cosine",
        },
    }
}


if es.indices.exists(index=index_name):
    print("exists!")
    pass
else:
    es.indices.create(index=index_name, body={"mappings": mapping, "settings": setting})

    data = []

    def appendbulk(row):
        targetstring = f"category: {row['category']}, author: {row['author']}, introduction: {row['introduction']}, title: {row['title']}"
        embedding = model.encode(targetstring)

        data.append({
            "_index": index_name,
            "_source": {
                "author": row["author"],
                "category": row["category"],
                "introduction": row["introduction"],
                "publisher": row["publisher"],
                "title": row["title"],
                "publish_date": row["publish_date"],
                "isbn": row["isbn"],
                "toc": row["toc"],
                "embedding": embedding,
            },
        })


with pd.read_csv(input_filename, chunksize=chunksize, encoding="utf-8") as reader:
    for chunk in tqdm(reader):
        print("--------------------------------------")
        chunk.apply(appendbulk, axis=1)
        bulk(es, data)
        data.clear()
