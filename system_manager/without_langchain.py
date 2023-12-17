import os
import re

# api keys go here
import keys

OPENAI_API_KEY = keys.OPENAI_API_KEY
HUGGINGFACEHUB_API_TOKEN = keys.HUGGINGFACEHUB_API_TOKEN
naver_client_id = keys.naver_client_id
naver_client_secret = keys.naver_client_secret
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
import openai
import elasticsearch
import threading
import urllib
import requests

# es=Elasticsearch([{'host':'localhost','port':9200}])
# es.sql.query(body={'query': 'select * from global_res_todos_acco...'})
import queue
import logging
import json

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import datetime

from elasticsearch_util.elasticsearch_retriever import ElasticSearchBM25Retriever
from utils.translation import translate_text

retriever = ElasticSearchBM25Retriever()
with open("/config.json") as f:
    config = json.load(f)


def getUserIntention(user_input):
    url = config["intention_generation_url"]
    data = {"input": user_input}
    intention_rawstring = requests.post(url, json=data)
    start = intention_rawstring.find("단서들:[") + len("단서들:[")
    end = intention_rawstring.find("]", start)
    clue_str = intention_rawstring[start:end]
    clues_list = clue_str.split(", ")

    title = None
    author = None
    publisher = None
    keyword_list = list()
    for pair in clues_list:
        try:
            word, classification = pair.split(":")
            if classification == "제목":
                title = word
            if classification == "저자":
                author = word
            if classification == "출판사":
                publisher = word
            if classification == "키워드":
                keyword_list.append(word)
        except ValueError:
            print("':' not found in the string adding to keyword_list")
            keyword_list.append(word)
    return title, author, publisher, keyword_list


def similar_booksearch(bookname, user_query) -> list:
    title_result = retriever.search_with_title(bookname)
    return_result = list()
    if len(title_result) == 0:
        return_result = keyword_search(user_query)
    elif len(title_result) == 1:
        return_result = retriever.knn_only_search(title_result[0].tensor)
    return return_result


def author_search(author_name, user_query) -> list:
    author_result = retriever.search_with_author(author_name)
    if len(author_result) == 0:
        author_result = keyword_search(user_query)
    return author_result


def publisher_search(publisher_name, user_query) -> list:
    publisher_result = retriever.search_with_publisher(publisher_name)
    if len(publisher_result) == 0:
        publisher_result = keyword_search(user_query)
    return publisher_result


def keyword_search(user_query) -> list:
    # TODO
    result = retriever.search_with_query(user_query)
    return result


def evaluate_books(book_list) -> list:
    # TODO
    return


def generate_recommendation_sentence(book_list, user_query):
    # TODO
    return


def generate_meta_search_sentence(book_list, user_query):
    # TODO
    return


def interact_opensourceGeneration(
    webinput_queue, weboutput_queue, langchoice_queue, user_id
):
    # region logging setting
    log_file_path = f"log_from_user_{user_id}.log"

    # logger for each thread
    logger = logging.getLogger(f"UserID-{user_id}")
    logger.setLevel(logging.INFO)

    # file handler for each thread
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter(
        "%(asctime)s [%(threadName)s - %(thread)s] %(levelname)s: %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # endregion
    elasticsearch_url = config["elasticsearch_url"]
    retriever = ElasticSearchBM25Retriever(
        elasticsearch.Elasticsearch(
            elasticsearch_url,
            verify_certs=False,
        ),
        "data",
    )
    print("start interact!")
    # region mongodb setting
    client = MongoClient(config["mongodb_uri"], server_api=ServerApi("1"))
    # endregion
    chatturn = 0
    while 1:
        webinput = webinput_queue.get()
        langchoice = langchoice_queue.get()
        input_query = webinput
        web_output = None
        print("GETTING WEB INPUT")
        logger.warning(f"USERINPUT : {webinput}")
        webinput = translate_text("ko", webinput)

        title, author, publisher, keywordlist = getUserIntention(webinput)

        if title != None:
            search_result = similar_booksearch(title, webinput)
            passed_books = evaluate_books(search_result)
            generated_sentences = generate_recommendation_sentence(
                passed_books, input_query
            )
            weboutput_queue.put(generated_sentences)
        elif author != None:
            author_search(author, webinput)
        elif publisher != None:
            publisher_search(publisher, webinput)
        else:
            keyword_search(webinput)
        # region mongodb
        mongodoc = {
            "user_id": user_id,
            "usermsg": webinput,
            "aimsg": generated_sentences,
            "timestamp": datetime.datetime.now(),
            "turn": chatturn,
        }
        inserted_id = client.user_ai_interaction.interactions.insert_one(
            mongodoc
        ).inserted_id
        logger.warning(f"Interaction logged as docID : {inserted_id}")
        chatturn += 1
        # endregion
        print(f"PUTTING WEB OUTPUT in thread{threading.get_ident()}")
