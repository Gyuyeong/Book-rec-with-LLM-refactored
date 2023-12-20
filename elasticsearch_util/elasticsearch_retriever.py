import pandas as pd
from sentence_transformers import SentenceTransformer
import json
import elasticsearch
from typing import Type, List
import numpy as np

with open("config.json", "r", encoding="UTF-8") as f:
    config = json.load(f)


class Bookdata:
    """
    도서정보를 담고있는 클래스
    title, author, publisher, introduction, isbn, tensor
    로 인스턴스 생성
    str로 사용시 "title : 제목, author : 작가, introduction : 책 소개" 형태
    """

    def __init__(self, title, introduction, author, publisher, isbn, tensor):
        self.title = title
        self.author = author
        self.publisher = publisher
        self.introduction = introduction
        self.isbn = isbn
        self.tensor = tensor

    def __str__(self):
        return f"title : {self.title}, author : {self.author}, introduction : {self.introduction}"


class ElasticSearchBM25Retriever:
    """
    엘라스틱서치 검색 함수와 클라이언트 설정하는 클래스.
    엘라스틱서치 클라이언트와 인덱스명으로 클래스 인스턴스 생성.
    """

    def __init__(self, client, index_name: str):
        self.client = client
        self.index_name = index_name

    def search_with_query(self, query: str) -> List[Bookdata]:
        with open("config.json", encoding="UTF-8") as f:
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
            "size": n,
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
                np.array(r["_source"]["embedding"]),
            )
            docs.append(bd)

        return docs

    def search_with_author(self, query: str) -> List[Bookdata]:
        query_dict: dict()
        query_dict = {"query": {"term": {"author.keyword": query}}}
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
                np.array(r["_source"]["embedding"]),
            )
            docs.append(bd)

        print("\nfrom author book--------------------------------------------debug")
        print(docs)
        print("--------------------------------------------debug\n\n")
        return docs[0 : config["default_number_of_books_to_return"]]

    def search_with_title(self, query: str) -> List[Bookdata]:
        query_dict: dict()
        query_dict = {"query": {"term": {"title.keyword": {"value": query}}}}

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
                np.array(r["_source"]["embedding"]),
            )
            docs.append(bd)

        print("\nfrom title book--------------------------------------------debug")
        print(docs)
        print("--------------------------------------------debug\n\n")
        return docs[0 : config["default_number_of_books_to_return"]]

    def search_with_publisher(self, query: str) -> List[Bookdata]:
        query_dict: dict()
        query_dict = {"query": {"term": {"publisher.keyword": query}}}

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
                np.array(r["_source"]["embedding"]),
            )
            docs.append(bd)

        print("\nfrom pub book--------------------------------------------debug")
        print(docs)
        print("--------------------------------------------debug\n\n")
        return docs[0 : config["default_number_of_books_to_return"]]

    def knn_only_search(
        self, tensor: np.ndarray, excluded_title: str
    ) -> List[Bookdata]:
        with open("config.json", encoding="UTF-8") as f:
            config = json.load(f)
        n = config["elasticsearch_result_count"]
        query_dict = {
            "knn": {
                "field": "embedding",
                "query_vector": tensor.tolist(),
                "k": 10,
                "num_candidates": 50,
                "boost": 30,
            },
            "size": n,
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
                np.array(r["_source"]["embedding"]),
            )
            if bd.title != excluded_title:
                docs.append(bd)
        print("\nfrom knn only book--------------------------------------------debug")
        print(docs)
        for book in docs:
            print(book.title)
        print("--------------------------------------------debug\n\n")
        return docs
