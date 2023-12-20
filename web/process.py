import sys

sys.path.append(".")
from flask import Flask, render_template, request, session
from system_manager.without_langchain import interact_opensourceGeneration

from system_manager.with_langchain.with_langchain import interact_fullOpenAI

import threading
import queue
import uuid
import json

app = Flask(__name__)
app.secret_key = "12341234"  # temporary secret key
idthreadDict = {}
input_queue_dict = {}
langchoice_queue_dict = {}
output_queue_dict = {}
with open("config.json", encoding="UTF-8") as f:
    config = json.load(f)


def generate_user_id() -> str:
    """
    세션 접속하는 유저에게 uuid 생성
    """
    return str(uuid.uuid4())


@app.route("/demo")
def home():
    """
    유저 세션 접속시 해당 유저 전용 쓰레드 생성
    """
    global input_queue_dict
    global langchoice_queue_dict
    global output_queue_dict
    global idthreadDict
    if "user_id" in session:
        user_id = session["user_id"]
    else:
        # generate new user_id and store
        user_id = generate_user_id()
        session["user_id"] = user_id
        print(f"process got userid of : {user_id}")
        print(f"new session for user : {user_id}")

    input_queue_dict[user_id] = queue.Queue()
    output_queue_dict[user_id] = queue.Queue()
    langchoice_queue_dict[user_id] = queue.Queue()

    # start server-side loop in separate thread

    if config["flowchoice"] == "langchain":
        server_thread = threading.Thread(
            target=interact_fullOpenAI,
            args=(
                input_queue_dict[user_id],
                output_queue_dict[user_id],
                langchoice_queue_dict[user_id],
                user_id,
            ),
        )
    if config["flowchoice"] == "nolangchain":
        server_thread = threading.Thread(
            target=interact_opensourceGeneration,
            args=(
                input_queue_dict[user_id],
                output_queue_dict[user_id],
                langchoice_queue_dict[user_id],
                user_id,
            ),
        )
    server_thread.daemon = True
    server_thread.start()
    print(f"thread id {server_thread} started for user {user_id}")
    idthreadDict[user_id] = server_thread
    return render_template("index_mobile.html", user_id=user_id)


@app.route("/process", methods=["POST"])
def process():
    """
    위에서 생성한 쓰레드에 queue로 웹에서의 유저 입력을 전달
    """
    global input_queue_dict
    global langchoice_queue_dict
    global output_queue_dict
    global idthreadDict
    print("in process")
    # check user_id already in session
    if "user_id" in session:
        user_id = session["user_id"]
    else:
        # generate new user_id and store
        # should not happen basically
        user_id = generate_user_id()
        session["user_id"] = user_id

    input_data = request.form["inputField"]
    print(f"user input : {input_data}")
    # put user input into input queue
    input_queue_dict[user_id].put(input_data)
    lang_choice = request.form["dropdown"]
    print(f"model choice : {lang_choice}")
    langchoice_queue_dict[user_id].put(lang_choice)

    # wait output from the server-side loop

    output = output_queue_dict[user_id].get()

    return output


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
