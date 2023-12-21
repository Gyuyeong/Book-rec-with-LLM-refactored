from sentence_transformers import SentenceTransformer
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm

# initializing the model for embedding
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# Path to the input CSV file
input_filename = f"csv file directory to upload"

# Size of chunks to read from the CSV file
chunksize = 50

# Name of the Elasticsearch index
index_name = "data"

# Elasticsearch client setup with connection details
with open("config.json", "r", encoding="UTF-8") as f:
    import json

    config = json.load(f)
es = Elasticsearch(
    ["https://your_elasticsearch_server:port"],
    basic_auth=("username", "password"),
    verify_certs=False,
    timeout=30,
    max_retries=10,
    retry_on_timeout=True,
)

# Setting up analysis and tokenization for Korean language using Nori
setting = {
    "analysis": {
        "analyzer": {
            "nori_analyzer": {
                "type": "nori",
                "tokenizer": "nori_mixed",
            }
        },
        "tokenizer": {
            "nori_mixed": {
                "type": "nori_tokenizer",
                "decompound_mode": "mixed",
            }
        },
    },
}

# Mapping for the data structure in Elasticsearch
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

# Check if the index exists, if not, create a new index with the specified mapping and settings
if es.indices.exists(index=index_name):
    print("exists!")
    pass
else:
    es.indices.create(index=index_name, body={"mappings": mapping, "settings": setting})

# List to store data to be bulk uploaded
data = []


# Function to append data to bulk list
def appendbulk(row):
    # Concatenating category, author, introduction title for embedding
    targetstring = f"category: {row['category']}, author: {row['author']}, introduction: {row['introduction']}, title: {row['title']}"

    # Generating embedding for the targetstring
    embedding = model.encode(targetstring)

    # Appending data in the required format for Elasticsearch
    data.append(
        {
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
        }
    )


# Reading the CSV file in chunks
with pd.read_csv(input_filename, chunksize=chunksize, encoding="utf-8") as reader:
    for chunk in tqdm(reader):
        print("--------------------------------------")

        # Applying 'appendbulk' function to each row in the chunk
        chunk.apply(appendbulk, axis=1)

        # Bulk uploading the data to Elasticsearch
        bulk(es, data)

        # Clearing the list for the next chunk
        data.clear()
