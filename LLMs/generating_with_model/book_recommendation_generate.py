import sys

sys.path.append(".")
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from utils.translation import translate_text
from peft import PeftModel

IGNORE_INDEX = -100
PAD_TOKEN = "[PAD]"
EOS_TOKEN = "</s>"
BOS_TOKEN = "</s>"
UNK_TOKEN = "</s>"
# 원활한 생성을 위해 추가한 custom token들
CLUE_TOKEN = "<CLUE>"
REASONING_TOKEN = "<REASONING>"
LABEL_TOKEN = "<LABEL>"

PROMPT_DICT = {
    "intention": (
        "이 모델은 '입력 문장'을 분석해서 의도를 분류하는 모델입니다. "
        "분류할 의도로는 [메타 정보 검색], [키워드 검색], [유사도서 검색], [판매량 검색], 그리고 [그 외]가 있습니다. "
        "입력 문장이 책 추천과 관련이 없는 경우 [그 외]로 분류합니다. "
        "입력 문장이 책 추천과 관련이 있는 경우, 다음과 같이 분류합니다. "
        "저자, 출판사, 기간 등 메타 정보가 포함된 경우 [메타 정보 검색]으로 분류합니다. "
        "키워드 정보만 포함된 경우 [키워드 검색]으로 분류합니다. "
        "제목 정보가 포함된 경우 예외 없이 [유사도서 검색]으로 분류합니다. "
        "판매량 정보가 들어간 경우 예외 없이 [판매량 검색]으로 분류합니다. "
        "우선, 입력 데이터를 받으면 주어진 입력 데이터에서 의도를 분류할 때 도움이 될 수 있는 단서들을 추출합니다. "
        "그리고 각 단서마다 어떤 종류의 단서인지 표시합니다. "
        "그 다음, 주어진 입력 데이터와 단서들을 바탕으로 어떤 의도인지 추론하는 글을 생성합니다. "
        "그 후, 입력 데이터와 단서들과 추론한 글을 바탕으로 주어진 5가지 의도 중 하나를 생성합니다. [의도: ]를 적고 생성합니다. "
        "반드시 [단서들], [추론], [의도] 순서대로 생성해야 합니다. "
        "의도를 생성할 때 반드시 주어진 글자들과 동일하게 생성해야 합니다.\n"
        "입력: {input}\n\n"
    ),
    "evaluation": (
        "EVALUATION: 주어진 사용자의 질문과 정보를 바탕으로 한 책 추천의 적절성을 평가해 주세요. "
        "만약 추천된 책이 사용자의 QUERY과 관련되어 있고 INFO에 잘 부합한다면 'Pass'을, 그렇지 않다면 'Fail'을 부여하세요. "
        "책의 주제, 사용자 질문과의 관련성, 그리고 사용자의 필요에 대한 적합성을 고려하세요. "
        "추천이 적절하고 관련이 있는지 평가하는 것이 중요합니다.\n{input}\n\n"
    ),
    "introduction": (
        "### Prompt(명령):\nresponse_to_user: {{input에 주어진 사용자 질의에 응답하는 문구를 생성해줘}}\n\n### Input(입력):{input}\n\n### Response(응답):"
    ),
    "generation": (
        "### Prompt(명령):\nbook_recommendation: {{input에 주어진 책 정보와 사용자 질의를 바탕으로 그 책을 추천한 사유를 생성해줘}}\n\n### Input(입력):{input}\n\n### Response(응답):"
    ),
}


# MODEL_ID = "EleutherAI/polyglot-ko-12.8b"
MODEL_ID = "rycont/kakaobrain__kogpt-6b-8bit"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, padding_side="right", model_max_length=512
)
tokenizer.add_special_tokens(
    {"eos_token": EOS_TOKEN, "bos_token": BOS_TOKEN, "unk_token": UNK_TOKEN}
)
tokenizer.pad_token = tokenizer.eos_token

tokenizer.add_tokens([CLUE_TOKEN, REASONING_TOKEN, LABEL_TOKEN], special_tokens=True)
model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(
    model=model,
    model_id="./consolidated_models/kakao_models/n_100_lr_3e_5/checkpoint-9300/",
)


import re


def generate_recommendation(
    user_input: str,
    book_data: list,
    target_lang,
    isbn_list: list,
    model=model,
    tokenizer=tokenizer,
):
    outstring = str()
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    generation_args = dict(
        num_beams=2,
        repetition_penalty=2.0,
        no_repeat_ngram_size=4,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.15,
        early_stopping=True,
    )

    mapped_prompt = PROMPT_DICT["introduction"].format_map({"input": user_input})
    response = generator(mapped_prompt, **generation_args)
    result = (response[0]["generated_text"]).replace(
        mapped_prompt, ""
    )  # response to query
    print(result)
    outstring = result + "<br>"
    # book dataset of book data
    list_book = [
        PROMPT_DICT["generation"].format_map(
            {"input": "user_query: {" + user_input + "}, " + book}
        )
        for book in book_data
    ]
    list_result = generator(list_book, **generation_args)

    pattern = r"title:\s*\[([^]]+)\],\s*author:\s*\[([^]]+)\]"

    for book_prompt, result, isbn in zip(list_book, list_result, isbn_list):
        title_and_author_result = re.findall(pattern, book_prompt)
        print(
            "["
            + title_and_author_result[0][0]
            + "] ("
            + title_and_author_result[0][1]
            + ")"
        )
        outstring += (
            "["
            + title_and_author_result[0][0]
            + "] ("
            + title_and_author_result[0][1]
            + ")"
            + "<br>"
        )
        final_result = result[0]["generated_text"].replace(book_prompt, "")
        print(final_result)
        print()
        final_result = translate_text(target_lang, final_result)
        outstring += (
            final_result
            + '<br><a href="https://www.booksonkorea.com/product/'
            + isbn
            + '" target="_blank" class="quickViewButton">Quick View</a><br><br>'
        )
    return outstring


# generate_recommendation(
#     model=model,
#     tokenizer=tokenizer,
#     user_input="게임과 관련된 라이트노벨 또는 만화 추천해줘",
#     book_data=[
#         "book: {title: [소드 아트 온라인 1(J 노블(J Novel))], author: [카와하라 레키], introduction: [베타테스터 시절부터 여기저기 플래그를 세우고 다니는 키리토와 그의 아내로  길드의 부단장을 역임했던 ‘섬광’ 아스나의 달달한 사랑이야기는 보너스~!! 본편에서 더 많은 출연을 원했던 등장인물들이 쏟아져 나와 자신들의 이야기를 들려준다!! SAO팬이라면 누구나 고개를 끄덕이며 공감 100%!! SAO을 즐기는 또 다른 방식!! 4컷 만화로 누리는 또 다른 즐거움!!]}",
#         "book: {title: [오버로드 1: 불사자의 왕], author: [마루야마 쿠가네], introduction: [일본 연재 사이트에서 1,000만 조회수를 상회한 인기작 『오버로드』 제1권. 갑자기 새로운 세계에 떨어진 주인공이 어떻게 그 상황을 하나하나 대처해나가는지를 세밀하게 보여준다. 게임 위그드라실의 서비스 종료를 앞둔 밤. ‘아인즈 울 고운’의 길드장이자 ‘나자릭 지하대분묘’의 주인인 언데드 매직 캐스터 ‘모몬가’는 게임의 종료와 동시에 길드 아지트인 나자릭 지하대분묘 전체가 이세계로 전이한 것을 깨닫게 된다. NPC들은 자신만의 개성을 얻어 살아 움직이고, 모몬가는 더 이상 이것이 ‘게임’이 아니라 ‘또 다른 세상’이었는데….]}",
#     ],
# )

from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/generate/final", methods=["POST"])
def generate():
    data = request.json
    return generate_recommendation(
        data["input"], data["books"], data["lang"], data["isbn_list"]
    )


if __name__ == "__main__":
    app.run(port=5002)
