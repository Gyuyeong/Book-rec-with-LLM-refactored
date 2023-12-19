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
import random
import openai
import elasticsearch
import threading
import urllib
import requests
from utils.translation import translate_text

# modified langchain.chat_models ChatOpenAI
from system_manager.with_langchain.modifiedLangchainClass.openai import ChatOpenAI

from langchain import LLMChain


from langchain.tools import BaseTool

from langchain.agents import initialize_agent
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor

from langchain.memory import ConversationBufferMemory

import queue
import logging
import json

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import datetime

from LLMs.GPT_API_utils.Recommendation_Evaluation import isbookPass
from LLMs.GPT_API_utils.Generate_Recommendation import generate_recommendation

from elasticsearch_util.elasticsearch_retriever import ElasticSearchBM25Retriever

toolList = ["booksearch", "cannot", "elastic"]


def interact_fullOpenAI(webinput_queue, weboutput_queue, langchoice_queue, user_id):
    langchoice_Reference = {"en": " Answer in English.", "ko": " 한국어로 답변해줘."}
    chatturn = 0
    recommended_isbn = list()

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

    # region setting
    with open("config.json", "r", encoding="UTF-8") as f:
        config = json.load(f)
    # region mongodb setting
    client = MongoClient(config["mongodb_uri"], server_api=ServerApi("1"))
    # endregion
    web_output: str
    input_query: str

    elasticsearch_url = config["elasticsearch_url"]
    retriever = ElasticSearchBM25Retriever(
        elasticsearch.Elasticsearch(
            elasticsearch_url,
            verify_certs=False,
        ),
        config["elasticsearch_index_name"],
    )

    # endregion
    # region tool definition

    # tool that performs meta search
    class booksearch_Tool(BaseTool):
        name = "booksearch"
        description = (
            "Use this tool when searching based on brief information about a book you have already found. "
            "Use this tool to get simple information about books. "
            "You should be conservative when you judge whether user's request is a daily conversation or a request for book search. "
            "Only when it is about book search, use this tool. "
            "This tool searches book's title, author, publisher and isbn. "
            "Input to this tool can be single title, author, or publisher. "
            "You need to state explicitly what you are searching by. If you are searching by an author, use author: followed by the name of the book's author. If you are searching by a publisher, use publisher: followed by the name of the book's publisher. And if you are searching by the title, use title: followed by the name of the book's title."
            "The format for the Final Answer should be (number) title : book's title, author :  book's author, pubisher :  book's publisher. "
        )

        def _run(self, query: str):
            print("\nbook_search")
            if "author: " in query:
                print("\n=====author=====")
                result = retriever.search_with_author(query)
            elif "publisher: " in query:
                print("\n=====publisher=====")
                result = retriever.search_with_publisher(query)
            elif "title: " in query:
                print("\n=====title=====")
                result = retriever.search_with_title(query)

            return f"{result} I should give final answer based on these information. "

        def _arun(self, query: str):
            raise NotImplementedError("This tool does not support async")

    # tool that says cannot perform task
    class cannot_Tool(BaseTool):
        name = "cannot"
        description = (
            "Use this tool when there are no available tool to fulfill user's request. "
            "Do not enter this tool when user's request is about daily conversation."
        )

        def _run(self, query: str):
            result = "Cannot perform task. "
            print(result)

            # 강제 출력하려면 주석해제
            # nonlocal web_output
            # web_output = result
            result += "Thought:Couldn't perform task. I must inform user.\n"
            result += "Final Answer: "

            return result

        def _arun(self, query: str):
            raise NotImplementedError("This tool does not support async")

    # tool that performs query search
    class elastic_Tool(BaseTool):
        name = "elastic"
        default_num = config["default_number_of_books_to_return"]
        description = (
            "Use this tool only for recommending books to users. "
            "You should be conservative when you judge whether user's request is a daily conversation or a request for book recommendation. "
            "Only when it is about book recommendation, use this tool. "
            f"Format for Action input: (query, number of books to recommend) if specified, otherwise (query, {default_num})."
            "Final Answer format: (number) title: [Book's Title], author: [Book's Author], publisher: [Book's Publisher]."
            "Input may include the year."
        )

        # 파파고 번역을 위해 특수문자 제거
        def extract_variables(self, input_string: str):
            input_string = input_string.replace('"', "")
            variables_list = input_string.strip("()\n").split(", ")
            name = variables_list[0]
            num = int(variables_list[1])
            return name, num

        # 이미 추천된 도서를 배제
        def filter_recommended_books(self, result):
            filtered_result = []
            for book in result:
                # 책의 ISBN이 이미 recommended_isbn에 있는지 확인합니다.
                if book.isbn not in [item["isbn"] for item in recommended_isbn]:
                    filtered_result.append(book)

                else:
                    print("\nalready recommended this book!")
                    print(book.title)
                    print("\n")
            return filtered_result

        # I must give Final Answer base
        def _run(self, query: str):
            elastic_input, num = self.extract_variables(query)
            ko_translated_input = translate_text("ko", elastic_input)
            nonlocal input_query
            nonlocal web_output
            nonlocal langchoice

            recommendList = list()
            recommendList.clear()
            bookList = list()
            bookList.clear()
            count = 0

            # 유저 쿼리를 번역해서 검색
            result = retriever.search_with_query(ko_translated_input)
            if config["filter_out_reccommended_books"]:
                result = self.filter_recommended_books(result)

            if config["use_gpt_api_for_eval"]:
                # 쓰레드 생성해서 동시에 평가
                bookresultQueue = queue.Queue()

                def book_pass_thread(userquery: str, bookinfo):
                    nonlocal bookresultQueue
                    if isbookPass(userquery, bookinfo):
                        bookresultQueue.put(bookinfo)
                    return

                threadlist = []
                for book in result:
                    t = threading.Thread(
                        target=book_pass_thread, args=(input_query, book)
                    )
                    threadlist.append(t)
                    t.start()

                for t in threadlist:
                    t.join()

                while not bookresultQueue.empty():
                    book = bookresultQueue.get()
                    recommendList.append(book)
                    bookList.append(
                        {
                            "author": book.author,
                            "publisher": book.publisher,
                            "title": book.title,
                            "isbn": book.isbn,
                        }
                    )
            else:
                url = config["evaluation_generation_url"]
                recommendList = list()
                for book in result:
                    # make json for model request
                    fullstring = (
                        f"QUERY: {{user_query}}, INFO: "
                        + f"{{title='{book.title}' introduction='{book.introduction}' author='{book.author}' publisher='{book.publisher}' isbn={book.isbn}}}"
                    )
                    data = {"fullstring": fullstring}
                    evaluate_rawstring = requests.post(url, json=data).text
                    if evaluate_rawstring == "Pass":
                        recommendList.append(book)
                        bookList.append(
                            {
                                "author": book.author,
                                "publisher": book.publisher,
                                "title": book.title,
                                "isbn": book.isbn,
                            }
                        )
            # 최종 출력을 위한 설명 만들기
            # 추천 결과가 유저가 요청한 추천수보다 많은 경우 or 유저가 따로 요청하지 않은 경우
            if len(recommendList) >= num or num == 3:
                reallength = min(len(recommendList), num)
                for i in range(reallength):
                    recommended_isbn.append(
                        {
                            "turnNumber": chatturn,
                            "author": recommendList[i].author,
                            "publisher": recommendList[i].publisher,
                            "title": recommendList[i].title,
                            "isbn": recommendList[i].isbn,
                        }
                    )
                if config["use_gpt_api_for_final"]:
                    result = ""
                    for i in range(reallength):
                        result += (
                            f"[{bookList[i]['title']}] ({bookList[i]['author']})<br>"
                        )
                        completion = generate_recommendation(
                            langchoice_Reference[langchoice],
                            recommendList[i],
                            input_query,
                        )
                        result += (
                            completion["choices"][0]["message"]["content"]
                            + '<br><a href="https://www.booksonkorea.com/product/'
                            + str(recommendList[i].isbn)
                            + '" target="_blank" class="quickViewButton">Quick View</a><br><br>'
                        )
                    web_output = result
                else:
                    url = config["final_generation_url"]
                    bookStringList = list()
                    isbnlist = list()
                    for book in recommendList:
                        fullstring = (
                            "book: "
                            + f"{{title=[{book.title}], author=[{book.author}], introduction=[{book.introduction}]}}"
                        )
                        bookStringList.append(fullstring)
                        isbnlist.append(book.isbn)
                    data = {
                        "input": ko_translated_input,
                        "books": bookStringList,
                        "isbn_list": isbnlist,
                        "lang": langchoice,
                    }
                    web_output = requests.post(url, json=data).text

                logger.info(f"web output set to {web_output}")
                return f"{bookList[0:num]}  "
            else:
                print(
                    f"smth went wrong: less then {num} pass found in thread{threading.get_ident()}"
                )

                return f"less then {num} pass found"

        def _arun(self, radius: int):
            raise NotImplementedError("This tool does not support async")

    tools = [elastic_Tool(), cannot_Tool(), booksearch_Tool()]

    prefix = """
    Have a conversation with a human, answering the following questions as best you can. 
    User may want some book recommendations, book search, or daily conversation. 
    You have access to the following tools:
    """
    suffix = """
    For daily conversation, please give user the Final Answer right away without using any tools. 
    It should be remembered that the current year is 2023. 
    You can speak Korean and English. 
    So when user wants the answer in Korean or English, you should give Final Answer in that language. 
    The name of the tool that can be entered into Action can only be elastic, cannot, and booksearch. 
    If the user asks for recommendation of books, you should answer with just title, author, and publisher. 
    You must finish the chain right after elastic tool is used. 
    Begin! 
    {chat_history}
    Question: {input}
    {agent_scratchpad}
    """

    # memory
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )

    memory = ConversationBufferMemory(memory_key="chat_history")

    llm_chain = LLMChain(llm=ChatOpenAI(temperature=0.4), prompt=prompt)

    agent = ZeroShotAgent(
        llm_chain=llm_chain,
        tools=tools,
        verbose=True,
    )

    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
    )
    # endregion

    while 1:
        webinput = webinput_queue.get()
        langchoice = langchoice_queue.get()
        input_query = webinput
        web_output = None
        print("GETTING WEB INPUT")
        logger.warning(f"USERINPUT : {webinput}")
        chain_out = agent_chain.run(input=webinput + langchoice_Reference[langchoice])
        print(f"PUTTING WEB OUTPUT in thread{threading.get_ident()}")
        # mongodb database name = user_ai_interaction & mongodb collection name = interactions
        if web_output is None:
            mongodoc = {
                "user_id": user_id,
                "usermsg": webinput,
                "aimsg": chain_out,
                "timestamp": datetime.datetime.now(),
                "turn": chatturn,
            }
            inserted_id = client.user_ai_interaction.interactions.insert_one(
                mongodoc
            ).inserted_id
            weboutput_queue.put(chain_out)
            logger.warning(f"OUTPUT : {chain_out}")
            logger.warning(f"Interaction logged as docID : {inserted_id}")
        else:
            mongodoc = {
                "user_id": user_id,
                "usermsg": webinput,
                "aimsg": web_output,
                "timestamp": datetime.datetime.now(),
                "turn": chatturn,
            }
            inserted_id = client.user_ai_interaction.interactions.insert_one(
                mongodoc
            ).inserted_id
            weboutput_queue.put(web_output)
            logger.warning(f"OUTPUT : {web_output}")
            logger.warning(f"Interaction logged as docID : {inserted_id}")
        chatturn += 1
