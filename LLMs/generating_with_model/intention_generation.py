from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from peft import PeftModel

IGNORE_INDEX = -100
PAD_TOKEN = "[PAD]"
EOS_TOKEN = "</s>"
BOS_TOKEN = "</s>"
UNK_TOKEN = "</s>"
CLUE_TOKEN = "<CLUE>"
REASONING_TOKEN = "<REASONING>"
LABEL_TOKEN = "<LABEL>"
PROMPT_DICT = {
    "prompt_input": (
        "이 모델은 '입력 문장'을 분석해서 의도를 분류하는 모델입니다. "
        + "분류할 의도로는 '메타 정보 검색', '키워드 검색', '유사도서 검색', '판매량 검색', 그리고 '그 외'가 있습니다. "
        + "입력 문장이 책 추천과 관련이 없는 경우, '그 외'로 분류합니다. "
        + "입력 문장이 책 추천과 관련이 있는 경우, 다음과 같이 분류합니다. "
        + "저자, 출판사, 기간 등 메타 정보가 포함된 경우 '메타 정보 검색'으로 분류합니다. "
        + "키워드 정보만 포함된 경우 '키워드 검색'으로 분류합니다. "
        + "제목 정보가 포함된 경우 예외 없이 '유사도서 검색'으로 분류합니다. "
        + "판매량 정보가 들어간 경우 예외 없이 '판매량 검색'으로 분류합니다. "
        + "우선, 입력 데이터를 받으면 주어진 입력 데이터에서 의도를 분류할 때 도움이 될 수 있는 단서들을 추출합니다. "
        + "그리고 각 단서마다 어떤 종류의 단서인지 추가합니다. "
        + "그 다음, 주어진 입력 데이터와 단서들을 바탕으로 어떤 의도인지 추론하는 글을 생성합니다. "
        + "그 후, 입력 데이터와 단서들과 추론한 글을 바탕으로 의도를 분류합니다. "
        + "반드시 '단서들', '추론', '의도' 순서대로 생성해야 합니다. "
        + "의도를 추론할 때 반드시 주어진 의도를 문자 그대로 생성해야 합니다.\n"
        "입력: {input}\n\n"
    )
}

model = AutoModelForCausalLM.from_pretrained("rycont/kakaobrain__kogpt-6b-8bit")
tokenizer = AutoTokenizer.from_pretrained(
    "rycont/kakaobrain__kogpt-6b-8bit", padding_size="right", model_max_length=1024
)
tokenizer.add_special_tokens(
    {
        "eos_token": EOS_TOKEN,
        "bos_token": BOS_TOKEN,
        "unk_token": UNK_TOKEN,
    }
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_tokens([CLUE_TOKEN, REASONING_TOKEN, LABEL_TOKEN], special_tokens=True)
model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(
    model=model,
    model_id="./LLMs/actual_models/intention/",
)


def generate_intention(user_input: str, model=model, tokenizer=tokenizer):
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

    generation_args = dict(
        num_beams=2,
        repetition_penalty=2.0,
        no_repeat_ngram_size=4,
        max_new_tokens=1024,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.1,
        early_stopping=True,
        temperature=1.0,
    )

    mapped_prompt = PROMPT_DICT["prompt_input"].format_map({"input": user_input})
    response = generator(mapped_prompt, **generation_args)
    result = (response[0]["generated_text"]).replace(mapped_prompt, "")
    return result


from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    return generate_intention(data["input"])


if __name__ == "__main__":
    app.run(port=5001)
