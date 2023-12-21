import os

# api keys go here
import keys

OPENAI_API_KEY = keys.OPENAI_API_KEY
HUGGINGFACEHUB_API_TOKEN = keys.HUGGINGFACEHUB_API_TOKEN
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
import elasticsearch
import threading
import requests
from typing import List

import queue
import logging
import json

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import datetime

from elasticsearch_util.elasticsearch_retriever import ElasticSearchBM25Retriever
from utils.translation import translate_text

with open("config.json", "r", encoding="UTF-8") as f:
    config = json.load(f)

elasticsearch_url = config["elasticsearch_url"]
print(elasticsearch_url)
retriever = ElasticSearchBM25Retriever(
    elasticsearch.Elasticsearch(
        elasticsearch_url,
        verify_certs=False,
    ),
    config["elasticsearch_index_name"],
)


def getUserIntention(user_input):
    """user_input : 유저 입력 문자열
    return : 단서에서 추출된 제목, 저자, 출판사, 키워드, 의도파악여부
    """
    url = config["intention_generation_url"]
    data = {"input": user_input}
    intention_rawstring = requests.post(url, json=data)
    # 생성된 intention에서 단서추출
    start = intention_rawstring.text.find("단서들: [") + len("단서들: [")
    end = intention_rawstring.text.find("]", start)
    clue_str = intention_rawstring.text[start:end]
    clues_list = clue_str.split(", ")

    title = None
    author = None
    publisher = None
    is_else = False
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
    if "의도: 그 외" in intention_rawstring.text:
        is_else = True
    return title, author, publisher, keyword_list, is_else


def similar_booksearch(bookname, user_query) -> List:
    """
    bookname : 유사도서 검색할 도서 title 문자열
    user query : 유저 쿼리 문자열
    return : Bookdata 의 리스트
    """
    title_result = retriever.search_with_title(bookname)
    return_result = list()
    if len(title_result) == 0:
        return_result = keyword_search(user_query)
    else:
        return_result = retriever.knn_only_search(title_result[0].tensor, bookname)
    return return_result


def author_search(author_name, user_query):
    """
    author_name : 유사도서 검색할 도서 작가 문자열
    user query : 유저 쿼리 문자열
    return : Bookdata 의 리스트, 작가검색 여부(실패후 키워드 검색 했을시 True)
    """
    author_result = retriever.search_with_author(author_name)
    did_keyword_search = False
    if len(author_result) == 0:
        author_result = keyword_search(user_query)
        did_keyword_search = True
    return author_result, did_keyword_search


def publisher_search(publisher_name, user_query):
    """
    publisher_name : 유사도서 검색할 도서 작가 문자열
    user query : 유저 쿼리 문자열
    return : Bookdata 의 리스트, 작가검색 여부(실패후 키워드 검색 했을시 True)
    """
    publisher_result = retriever.search_with_publisher(publisher_name)
    did_keyword_search = False
    if len(publisher_result) == 0:
        publisher_result = keyword_search(user_query)
        did_keyword_search = True
    return publisher_result, did_keyword_search


def keyword_search(user_query) -> List:
    """
    user query : 유저 쿼리 문자열
    return : Bookdata 의 리스트
    """
    result = retriever.search_with_query(user_query)
    return result


def evaluate_books(book_list, user_query) -> List:
    """
    book_list : 평가할 Bookdata의 리스트
    user_query : 유저 쿼리
    return : 최대 설정된 반화도서 수 만큼의 Bookdata의 리스트
    """
    if config["use_gpt_api_for_eval"]:
        from LLMs.GPT_API_utils.Recommendation_Evaluation import isbookPass

        # gpt api의 경우 쓰레드 생성해서 동시에 평가
        bookresultQueue = queue.Queue()

        def book_pass_thread(userquery: str, bookinfo):
            nonlocal bookresultQueue
            if isbookPass(userquery, bookinfo):
                bookresultQueue.put(bookinfo)
            return

        threadlist = []
        for book in book_list:
            t = threading.Thread(target=book_pass_thread, args=(user_query, book))
            threadlist.append(t)
            t.start()

        for t in threadlist:
            t.join()

        returnlist = list()
        while not bookresultQueue.empty():
            returnlist.append(bookresultQueue.get())
        return returnlist[0 : config["default_number_of_books_to_return"]]

    else:
        url = config["evaluation_generation_url"]
        passlist = list()
        for book in book_list:
            fullstring = (
                f"QUERY: {{user_query}}, INFO: "
                + f"{{title='{book.title}' introduction='{book.introduction}' author='{book.author}' publisher='{book.publisher}' isbn={book.isbn}}}"
            )
            data = {"fullstring": fullstring}
            evaluate_rawstring = requests.post(url, json=data).text
            if evaluate_rawstring == "Pass":
                passlist.append(book)
        return passlist[0 : config["default_number_of_books_to_return"]]


def generate_recommendation_sentence(book_list, user_query, langchoice):
    """
    book_list : 추천사유 생성할 Bookdata의 리스트
    user_query : 유저 쿼리
    langchoice : 선택된 언어 코드(문자열); "ko" 또는 "en"
    return : 웹에 출력할 html
    """
    if config["use_gpt_api_for_eval"]:
        langchoice_Reference = {"en": " Answer in English.", "ko": " 한국어로 답변해줘."}
        from LLMs.GPT_API_utils.Generate_Recommendation import generate_recommendation

        result = str()
        for book in book_list:
            result += f"[{book.title}] ({book.author})<br>"
            completion = generate_recommendation(
                langchoice_Reference[langchoice], book, user_query
            )
            result += (
                completion["choices"][0]["message"]["content"]
                + '<br><a href="https://www.booksonkorea.com/product/'
                + str(book.isbn)
                + '" target="_blank" class="quickViewButton">Quick View</a><br><br>'
            )
        return result
    else:
        url = config["final_generation_url"]
        bookStringList = list()
        isbnlist = list()
        for book in book_list:
            fullstring = (
                "book: "
                + f"{{title=[{book.title}], author=[{book.author}], introduction=[{book.introduction}]}}"
            )
            bookStringList.append(fullstring)
            isbnlist.append(book.isbn)
        data = {
            "input": user_query,
            "books": bookStringList,
            "isbn_list": isbnlist,
            "lang": langchoice,
        }
        final_rawstring = requests.post(url, json=data).text
        return final_rawstring


def generate_meta_search_sentence(book_list, user_query, langchoice):
    """
    book_list : 메타정보 검색결과 생성할 Bookdata의 리스트
    user_query : 유저 쿼리(현재는 사용 안함)
    langchoice : 선택된 언어 코드(문자열); "ko" 또는 "en"
    return : 웹에 출력할 html
    """
    returnstring = str()
    if langchoice == "ko":
        returnstring = config["meta_search_canned_text_ko"]
        for book in book_list:
            returnstring += (
                "<br>"
                + f"제목 : {book.title} <br>"
                + f"작가 : {book.author} <br>"
                + f"출판사 : {book.publisher} <br>"
                + "<br>"
                + '<br><a href="https://www.booksonkorea.com/product/'
                + str(book.isbn)
                + '" target="_blank" class="quickViewButton">Quick View</a><br><br>'
            )
    elif langchoice == "en":
        returnstring = config["meta_search_canned_text_en"]
        for book in book_list:
            returnstring += (
                "<br>"
                + f"Title : {book.title} <br>"
                + f"Author : {book.author} <br>"
                + f"Publisher : {book.publisher} <br>"
                + "<br>"
                + '<br><a href="https://www.booksonkorea.com/product/'
                + str(book.isbn)
                + '" target="_blank" class="quickViewButton">Quick View</a><br><br>'
            )
    return returnstring


def interact_opensourceGeneration(
    webinput_queue, weboutput_queue, langchoice_queue, user_id
):
    """
    webinput_queue : 이 세션 유저의 쿼리를 받는 queue
    weboutput_queue : 이 세션 유저의 웹에 출력할 html을 담는 queue
    langchoice_queue : 이 세션 유저가 요청한 언어를 담는 queue
    user_id : 이 세션 유저 uuid (세션 생성시 uuid 생성)
    """
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

        title, author, publisher, keywordlist, is_else = getUserIntention(webinput)
        generated_sentences = str()
        # 의도가 '그 외' 인 경우
        if is_else:
            weboutput_queue.put("Cannot handle intention")
            print("in else")
        # 유사도서 검색
        elif title != None:
            search_result = similar_booksearch(title, webinput)
            passed_books = evaluate_books(search_result, webinput)
            if len(passed_books) == 0:
                weboutput_queue.put(config["no_book_found_text"])
            else:
                generated_sentences = generate_recommendation_sentence(
                    passed_books, webinput, langchoice
                )
                weboutput_queue.put(generated_sentences)
        # 작가 검색
        elif author != None:
            search_result, did_keyword_search = author_search(author, webinput)
            print(search_result, did_keyword_search)
            # 작가 검색 실패시 키워드검색
            if did_keyword_search:
                passed_books = evaluate_books(search_result, webinput)
                if len(passed_books) == 0:
                    weboutput_queue.put(config["no_book_found_text"])
                else:
                    generated_sentences = generate_recommendation_sentence(
                        passed_books, webinput, langchoice
                    )
                    weboutput_queue.put(generated_sentences)
            # 작가 검색 성공
            else:
                generated_sentences = generate_meta_search_sentence(
                    search_result, webinput, langchoice
                )
                weboutput_queue.put(generated_sentences)
        # 출판사 검색
        elif publisher != None:
            search_result, did_keyword_search = publisher_search(publisher, webinput)
            # 출판사 검색 실패시 키워드검색
            if did_keyword_search:
                passed_books = evaluate_books(search_result, webinput)
                if len(passed_books) == 0:
                    weboutput_queue.put(config["no_book_found_text"])
                else:
                    generated_sentences = generate_recommendation_sentence(
                        passed_books, webinput, langchoice
                    )
                    weboutput_queue.put(generated_sentences)
            # 출판사 검색 성공
            else:
                generated_sentences = generate_meta_search_sentence(
                    search_result, webinput, langchoice
                )
                weboutput_queue.put(generated_sentences)
        # 키워드 검색
        else:
            search_result = keyword_search(webinput)
            passed_books = evaluate_books(search_result, webinput)
            if len(passed_books) == 0:
                weboutput_queue.put(config["no_book_found_text"])
            else:
                generated_sentences = generate_recommendation_sentence(
                    passed_books, webinput, langchoice
                )
                weboutput_queue.put(generated_sentences)
        print(generated_sentences)
        # region mongodb & logging
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
