from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from peft import PeftModel

# configuration
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "EVALUATION: 주어진 사용자의 질문과 정보를 바탕으로 한 책 추천의 적절성을 평가해 주세요. "
        "만약 추천된 책이 사용자의 QUERY과 관련되어 있고 INFO에 잘 부합한다면 'Pass'을, 그렇지 않다면 'Fail'을 부여하세요. "
        "책의 주제, 사용자 질문과의 관련성, 그리고 사용자의 필요에 대한 적합성을 고려하세요. "
        "추천이 적절하고 관련이 있는지 평가하는 것이 중요합니다.\n{input}\n\n"
    )
}

# get model and tokenizer
model = AutoModelForCausalLM.from_pretrained("rycont/kakaobrain__kogpt-6b-8bit")
tokenizer = AutoTokenizer.from_pretrained(
    "rycont/kakaobrain__kogpt-6b-8bit", padding_side="right", model_max_length=512
)
tokenizer.add_special_tokens(
    {
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
    }
)
tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(
    model=model,
    model_id="./LLMs/actual_models/evaluation/",
)


def generate_evaluation(fullstring: str, model=model, tokenizer=tokenizer):
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

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
    mapped_prompt = PROMPT_DICT["prompt_input"].format_map({"input": fullstring})
    response = generator(mapped_prompt, **generation_args)
    result = (response[0]["generated_text"]).replace(mapped_prompt, "")
    return result


from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/generate/evaluation", methods=["POST"])
def generate():
    data = request.json
    return generate_evaluation(data["fullstring"])


if __name__ == "__main__":
    app.run(port=5004)
