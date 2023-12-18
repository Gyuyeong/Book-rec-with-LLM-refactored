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
        "EVALUATION : input에 주어진 사용자 QUERY와 INFO를 비교하여 알맞은 책을 추천했는지 P 혹은 F로 평가해줘.\n\n "
        + "input: {input} \n\n"
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
        max_new_tokens=1024,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.5,
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
