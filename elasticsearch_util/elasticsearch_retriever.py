import pandas as pd
from sentence_transformers import SentenceTransformer
import json
import elasticsearch
from typing import Type
import numpy as np


class Bookdata:
    def __init__(self, title, author, publisher, introduction, isbn, tensor):
        self.title = title
        self.author = author
        self.publisher = publisher
        self.introduction = introduction
        self.isbn = isbn
        self.tensor = tensor


class ElasticSearchBM25Retriever:
    """
    To connect to an Elasticsearch instance that requires login credentials,
    including Elastic Cloud, use the Elasticsearch URL format
    https://username:password@es_host:9243. For example, to connect to Elastic
    Cloud, create the Elasticsearch URL with the required authentication details and
    pass it to the ElasticVectorSearch constructor as the named parameter
    elasticsearch_url.

    You can obtain your Elastic Cloud URL and login credentials by logging in to the
    Elastic Cloud console at https://cloud.elastic.co, selecting your deployment, and
    navigating to the "Deployments" page.

    To obtain your Elastic Cloud password for the default "elastic" user:

    1. Log in to the Elastic Cloud console at https://cloud.elastic.co
    2. Go to "Security" > "Users"
    3. Locate the "elastic" user and click "Edit"
    4. Click "Reset password"
    5. Follow the prompts to reset the password

    The format for Elastic Cloud URLs is
    https://username:password@cluster_id.region_id.gcp.cloud.es.io:9243.
    """

    def __init__(self, client, index_name: str):
        self.client = client
        self.index_name = index_name

    @classmethod
    def create(
        cls, elasticsearch_url: str, index_name: str, k1: float = 2.0, b: float = 0.75
    ) -> Type["ElasticSearchBM25Retriever"]:
        from elasticsearch import Elasticsearch

        # Create an Elasticsearch client instance
        es = Elasticsearch(
            ["https://115.71.239.131:9200"],
            basic_auth=("elastic", "HWH1rJdFReoOA8i-NPiy"),
            verify_certs=False,
        )

        # Define the index settings and mappings
        settings = {
            "analysis": {"analyzer": {"default": {"type": "standard"}}},
            "similarity": {
                "custom_bm25": {
                    "type": "BM25",
                    "k1": k1,
                    "b": b,
                }
            },
        }
        mappings = {
            "properties": {
                "content": {
                    "type": "text",
                    "similarity": "custom_bm25",
                }
            }
        }
        es.indices.create(index=index_name, mappings=mappings, settings=settings)
        return cls(es, index_name)

    def search_with_query(self, query: str) -> list[Bookdata]:
        with open("config.json") as f:
            config = json.load(f)
        n = config["elasticsearch_result_count"]
        # class Document(Serializable):
        # page_content: str
        # introduction : str
        # isbn : str
        model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
        embed = model.encode(query)
        query_dict = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "author",
                                    "category",
                                    "introduction",
                                    "publisher",
                                    "title",
                                    "toc",
                                ],
                                "boost": 1,
                            }
                        }
                    ]
                }
            },
            "knn": {
                "field": "embedding",
                "query_vector": embed,
                "k": 10,
                "num_candidates": 50,
                "boost": 30,
            },
            "size": 10,
        }
        res = self.client.search(
            index=self.index_name, body=query_dict, request_timeout=1200
        )
        docs = []

        for r in res["hits"]["hits"]:
            bd = Bookdata(
                r["_source"]["title"],
                r["_source"]["introduction"],
                r["_source"]["author"],
                r["_source"]["publisher"],
                r["_source"]["isbn"],
                np.array(r["_source"]["tensor"]),
            )
            docs.append(bd)

        return docs

    def search_with_author(self, query: str) -> list[Bookdata]:
        query_dict: dict()
        query_dict = {"query": {"term": {"author.keyword": {"value": query}}}}

        res = self.client.search(
            index=self.index_name, body=query_dict, request_timeout=1200
        )
        docs = []

        for r in res["hits"]["hits"]:
            docs.append(
                bd=Bookdata(
                    r["_source"]["title"],
                    r["_source"]["introduction"],
                    r["_source"]["author"],
                    r["_source"]["publisher"],
                    r["_source"]["isbn"],
                    np.array(r["_source"]["tensor"]),
                )
            )

        print("\nfrom_book--------------------------------------------debug")
        print(docs[0:2])
        print("--------------------------------------------debug\n\n")
        return docs[0:2]

    def search_with_title(self, query: str) -> list[Bookdata]:
        query_dict: dict()
        query_dict = {"query": {"term": {"title.keyword": {"value": query}}}}

        res = self.client.search(
            index=self.index_name, body=query_dict, request_timeout=1200
        )
        docs = []

        for r in res["hits"]["hits"]:
            docs.append(
                bd=Bookdata(
                    r["_source"]["title"],
                    r["_source"]["introduction"],
                    r["_source"]["author"],
                    r["_source"]["publisher"],
                    r["_source"]["isbn"],
                    np.array(r["_source"]["tensor"]),
                )
            )

        print("\nfrom_book--------------------------------------------debug")
        print(docs[0:2])
        print("--------------------------------------------debug\n\n")
        return docs[0:2]

    def search_with_publisher(self, query: str) -> list[Bookdata]:
        query_dict: dict()
        query_dict = {"query": {"term": {"publisher.keyword": {"value": query}}}}

        res = self.client.search(
            index=self.index_name, body=query_dict, request_timeout=1200
        )
        docs = []

        for r in res["hits"]["hits"]:
            docs.append(
                bd=Bookdata(
                    r["_source"]["title"],
                    r["_source"]["introduction"],
                    r["_source"]["author"],
                    r["_source"]["publisher"],
                    r["_source"]["isbn"],
                    np.array(r["_source"]["tensor"]),
                )
            )

        print("\nfrom_book--------------------------------------------debug")
        print(docs[0:2])
        print("--------------------------------------------debug\n\n")
        return docs[0:2]

    def knn_only_search(self, tensor: np.ndarray) -> list[Bookdata]:
        query_dict = {
            "knn": {
                "field": "embedding",
                "query_vector": tensor.tolist(),
                "k": 10,
                "num_candidates": 50,
                "boost": 30,
            },
            "size": 10,
        }
        res = self.client.search(
            index=self.index_name, body=query_dict, request_timeout=1200
        )
        docs = []
        for r in res["hits"]["hits"]:
            bd = Bookdata(
                r["_source"]["title"],
                r["_source"]["introduction"],
                r["_source"]["author"],
                r["_source"]["publisher"],
                r["_source"]["isbn"],
            )
            docs.append(bd)

        return docs
