# 생성형 모델

의도 분류, 추천 도서 평가, 도서 추천 사유 생성에 사용되는 생성형 모델에 관한 문서.

## Requirements
`Python 3.8.13` : 실험을 진행한 환경의 파이썬 버전.

`Python 3.10` 이상 버전의 경우 Google Colab에서 실행해보았을 때는 문제가 없었음. 다만 확실하지 않음

Packages
- `Pytorch 1.13.0+cu117` : 학습 데이터 로드 및 모델 학습을 위한 기능 지원
- `transformers 4.30.0` : Huggingface에서 오픈소스 LLM을 불러오기 및 fine-tuning
- `accelerate` : transformers에서 학습 시 요구
- `bitsandbytes` : 양자화된 모델 사용 및 모델에 양자화를 적용
- `peft`: LoRA 기법 적용

## Installation
```
pip install torch==1.13.0
pip install transformers==4.30.0
pip install -U accelerate
pip install bitsandbytes
pip install peft
```

### 설치 시 주의 사항
`Pytorch`와 `transformers`의 버전은 반드시 위 버전을 따라야 함. 학습 단계에서 torch 관련 변수들의 유무로 인해 학습이 실패하는 경우, 대부분 pytorch 버전 문제이다. `Transformers`가 현재로서는 `Pytorch 1.13.0` 버전만 호환이 잘 된다. 추후 업데이트가 있을 때까지는 `1.13.0` 버전을 사용해야 문제가 없다.

## Model Fine-tuning
`./training_model` 파일 내에 위치

- `book_evaluation_generation.py` : 추천 도서 평가 모델 학습 코드
- `book_recommendation_train.py` : 도서 추천 사유 생성 학습 코드
- `intention_train.py` : 의도 분류 학습 코드
- `consolidated_model_train.py` : 3가지 task 통합 모델

## Prompt
각 모델의 역할을 명시하는 글. 학습 데이터 앞에 붙여서 생성

각 task별 prompt :
```
PROMPT_DICT ={
    "intention": (
        "이 모델은 '입력 문장'을 분석해서 의도를 분류하는 모델입니다. "
        "분류할 의도로는 [메타 정보 검색], [키워드 검색], [유사도서 검색], [판매량 검색], 그리고 [그 외]가 있습니다. "
        "입력 문장이 책 추천과 관련이 없는 경우 [그 외]로 분류합니다. "
        "입력 문장이 책 추천과 관련이 있는 경우, 다음과 같이 분류합니다. "
        "저자, 출판사, 기간 등 메타 정보가 단서에 포함된 경우 [메타 정보 검색]으로 분류합니다. "
        "키워드 정보만 단서에 포함된 경우 [키워드 검색]으로 분류합니다. "
        "제목 정보가 단서에 포함된 경우 예외 없이 [유사도서 검색]으로 분류합니다. "
        "판매량 정보가 단서에 들어간 경우 예외 없이 [판매량 검색]으로 분류합니다. "
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
        "### Prompt(명령):\nbook_recommendation: {{input에 주어진 책 정보와 사용자 질의를 바탕으로 그 책을 추천한 사유를 생성해줘}}\n\n
        ### Input(입력):{input}\n\n### Response(응답):"
    ),
}
```

- `PROMPT_DICT["intention"]` : 의도 분류 prompt
- `PROMPT_DICT["evaluation"]` : 추천 도서 평가 prompt
- `PROMPT_DICT["introduction"]` : 추천 멘트 생성 prompt
- `PROMPT_DICT["generation"]` : 추천 사유 생성 prompt

각 prompt의 끝에 `{input}`에 입력 문장이 들어간다.

## Data Format For Training

데이터는 모두 JSON 형식으로 저장되어야 한다. Sample 및 학습에 사용한 데이터는 `./training_model/data`에 위치해 있다.

### 1. 의도 분류 모델
```
{
  "input": "입력 문장",
  "clues": "[단서들 ...]",
  "reasoning": "추론 문장",
  "intention": "의도"
}
```

- 입력 문장에는 사용자의 질의가 들어간다
- 단서는 대괄호 `"[]"` 안에 명시한다 (대괄호 옆에 따옴표도 반드시 표기). 입력 문장에서 단서로 쓰일만한 단어 또는 문구를 적고 바로 옆에 띄어쓰기 없이 `:` 을 적은 후, 해당 단서의 유형을 명시해준다.
  학습 데이터에 포함되어 있는 유형들은 다음과 같다:
  
  |유형|설명|예시|
  |---|---|---|
  |저자|책의 저자 정보|이원복:저자, 애거서 크리스티:저자|
  |출판사|책의 출판사 정보|민음사:출판사, 애플북스:출판사|
  |기간|책의 출판 시기와 연관된 정보|올해:기간, 3개월동안:기간|
  |제목|책의 제목|나의 라임 오랜지 나무:제목, 삼국지:제목|
  |판매량|판매량 정보가 필요한 정보|베스트셀러:판매량, 인기도서:판매량|
  |키워드|그 외 책 추천에 도움이 될만한 내용|초급 한국어:키워드, 학습:키워드|

  이외에도 필요에 따라 유형을 세부적으로 추가하는 것도 가능하다. 다만 추가할 경우 관련 유형을 담은 데이터를 많이 추가해줘야 한다.
- 추론은 단서를 바탕으로 template 형식으로 작성했다. 이는 최대한 같은 문장에 대해 일관성을 확보하기 위해서이다.
```
// 책 추천 관련 입력 문장
입력 문장은 책 추천을 원하고 있습니다. 이는 '[입력 문장에서 책 추천을 암시하는 부분]'를 통해서 확인할 수 있습니다. 단서를 보면, '[기간]' '[출판사]'에서 출판된, '[제목]'와/과 비슷한, 저자가 '[저자]'인, 판매량과 관련된 '[판매량 관련 단서]'인, '[키워드]'을/를 키워드로 하는 책을 원하고 있습니다. 따라서 입력 문장의 의도는 '[의도]'입니다.

// 책 추천과 무관한 입력 문장
입력 문장은 책 추천과 관련이 없습니다. [입력 문장의 목적을 간단하게 분석]. 따라서 입력 문장의 의도는 '그 외'입니다.
```

책 추천과 관련된 경우 단서에 적혀 있는 유형을 바탕으로 template의 각 빈칸을 채운다. 해당 유형이 없을 경우, 해당 부분을 제거한다.

- 의도는 5가지로 분류된다

  |의도|설명|
  |---|---|
  |메타 정보 검색|저자, 출판사, 출판 시기 등 도서의 메타 정보가 단서에 포함된 경우|
  |유사도서 검색|책 제목이 포함된 경우, 다른 유형의 단서들이 있더라도 유사도서 검색으로 분류|
  |판매량 검색|판매량 관련 단서가 포함된 경우, 위와 동일하다. 다만 책 제목과 충돌 시 유서도서 검색을 우선시한다|
  |키워드 검색|위 3가지 유형을 제외한 책 추천 관련 입력 문장|
  |그 외|책 추천과 관련이 없는 경우|

예시 데이터
```
{
        "input": "조지 오웰이 쓴 사회 비판적인 내용을 담은 책을 추천받고 싶어.",
        "clues": "[조지 오웰:저자, 사회 비판적인 내용:키워드]",
        "reasoning": "입력 문장은 책 추천을 원하고 있습니다. 이는 '책을 추천받고 싶어'를 통해서 확인할 수 있습니다. 단서를 보면, 저자가 '조지 오웰'인, '사회 비판적인 내용'을 키워드로 하는 책을 원하고 있습니다. 따라서 입력 문장의 의도는 '메타 정보 검색'입니다.",
        "intention": "메타 정보 검색"
}

{
        "input": "조선의 수도를 한양으로 정한 사람은?",
        "clues": "[조선의 수도:키워드, 한양으로 정한 사람:키워드]",
        "reasoning": "입력 문장은 책 추천과 관련이 없습니다. 조선의 수도를 정한 사람이 누구인지를 물어보는 것뿐입니다. 따라서 입력 문장의 의도는 '그 외'입니다.",
        "intention": "그 외"
}
```
### 2. 추천 도서 평가 모델
```
{
    "prompt": "EVALUATION: 주어진 사용자의 질문과 정보를 바탕으로 한 책 추천의 적절성을 평가해 주세요. 만약 추천된 책의 INFO가 사용자의 QUERY와 관련되어 있고 사용자의 요청에 잘 부합한다면 'Pass’를, 그렇지 않다면 'Fail'을 부여하세요. 책의 주제, 사용자 질문과의 관련성, 그리고 사용자의 필요와 의도에 대한 적합성을 고려하세요. 추천이 적절하고 관련이 있는지 평가하는 것이 중요합니다.\n\n",
    "input": "QUERY : {입력 문장}, INFO : {title='제목', introduction='책 소개', author='저자', publisher='출판사', isbn='ISBN'}",
    "Completion": "[Pass/Fail]"
}
```
예시 데이터
```
{
        "prompt": "EVALUATION: 주어진 사용자의 질문과 정보를 바탕으로 한 책 추천의 적절성을 평가해 주세요. 만약 추천된 책의 INFO가 사용자의 QUERY와 관련되어 있고 사용자의 요청에 잘 부합한다면 'Pass’를, 그렇지 않다면 'Fail'을 부여하세요. 책의 주제, 사용자 질문과의 관련성, 그리고 사용자의 필요와 의도에 대한 적합성을 고려하세요. 추천이 적절하고 관련이 있는지 평가하는 것이 중요합니다.\n\n",
        "input" : "QUERY : {조선후기 실학과 관련된 책 추천해줘}, INFO : {title='가자! 조선 실학: 수원 화성' introduction='「발도장 쿵쿵 역사시리즈」 제19권 『가자! 조선실학』은 조선 후기 성리학의 한계를 인정하고 사회 변화를 꿈꾸었던 실학에 대한 내용을 담고 있습니다. 먼저 실학이 발생한 조선 후기의 상황과 실학의 여러 주장들, 그리고 실학자를 등용한 정조에 관한 내용을 담았습니다. 그리고 ‘발도장 쿵쿵 현장탐방’ 부분에는 실학을 집대성하고, 정조의 개혁정신이 담긴 수원 화성과 화성행궁이 중심이 되어, 화성의 여러 건축물뿐 아니라 그 안에 담긴 새로운 기술과 기계 등이 서술되어 있습니다. 그리고 마지막 부분은 정조와 사도 세자의 능인 융·건릉과 유네스코 세계문화유산으로 지정된 수원 화성에 대해 담고 있습니다.' author='양대승' publisher='핵교' isbn=9788994591575}",
        "completion" : "Pass"
}

{
        "prompt": "EVALUATION: 주어진 사용자의 질문과 정보를 바탕으로 한 책 추천의 적절성을 평가해 주세요. 만약 추천된 책의 INFO가 사용자의 QUERY와 관련되어 있고 사용자의 요청에 잘 부합한다면 'Pass’를, 그렇지 않다면 'Fail'을 부여하세요. 책의 주제, 사용자 질문과의 관련성, 그리고 사용자의 필요와 의도에 대한 적합성을 고려하세요. 추천이 적절하고 관련이 있는지 평가하는 것이 중요합니다.\n\n",
        "input" : "QUERY : {조선후기 실학과 관련된 책 추천해줘}, INFO : {title='중국근세 경세사상연구' introduction='양명학이 출현하기 시작하는 성화, 홍치 연간(1465~1487)의 사상계보에 대해 개관하고, 특히 정주학의 한계를 극복하고자 노력한 당시 경세사상의 내용과 특징을 구준의 '대학연의보'를 중심으로 살펴본 연구서. 당시의 사상계보를 밝히고 더 나아가 남송공리학파와 명말청초 경세치용학과의 연계성을 모색, 중국 정치,학술 사상사에서 가장 중요한 위치를 차지하는 신유학의 발전과정과 중국 근세사회에서 주도적 역할을 담당한 사대부상을 이해하는 데 하나의 시금석이 되고 있다.' author='윤정분' publisher='혜안' isbn=9788984941632}",
        "completion" : "Fail"
}
```

Prompt 부분은 PROMPT_DICT 부분과 중복되기에 없어도 상관없다.
INFO 부분에 들어가는 내용은 `ElasticSearch`에서 검색한 결과물에서 입력 문장과 pass 또는 fail이 될만한 데이터를 선별해서 데이터를 만들었다. Pass의 경우에는 입력 문장과 잘 부합하는 책 정보, Fail의 경우에는 입력 문장과 잘 부합하지 않는 책 정보가 들어가 있으면 된다. Fail이 될 요인들 중 일부는 다음과 같다:
|유형|예시|
|---|---|
|아예 주제가 연관이 없는 경우|한국어 교재를 원하는데 책은 중국어 교재일 경우|
|입력 문장에서 원하는 수준의 책과 맞지 않은 경우|초금 한국어를 원하는데 고급 난이도일 경우|

추후 데이터를 추가할 시 여러 Fail의 요인들을 생각해서 관련 데이터를 다수 포함시켜야 할 것이다.
   
### 3. 책 추천 멘트 및 사유 생성 모델
```
{
    "prompt": "response_to_user: {input에 주어진 사용자 질의에 응답하는 문구를 생성해줘}",
    "input": "입력 문장",
    "completion": "추천 멘트\n\n"
}

{
    "prompt": "book_recommendation: {input에 주어진 책 정보와 사용자 질의를 바탕으로 그 책을 추천한 사유를 생성해줘}",
    "input": "user_query: {[입력 문장]}, book: {title: [제목], author: [저자], introduction: [책 소개]}",
    "completion": "- 추천 사유"
}
```
Prompt는 `PROMPT_DICT`와 중복되기는 하지만 두 데이터를 코드 상에서 구별하기 위해 보유하고 있다. 즉 여기서는 prompt를 데이터에 포함시켜 주는 것이 좋다. 

예시 데이터:

```
{
    "prompt": "response_to_user: {input에 주어진 사용자 질의에 응답하는 문구를 생성해줘}",
    "input": "슬픈 소설 책 추천해주세요! 그리고 인간관계에 대한 이야기도 추천해주세요! 아님 인간에 대한 책이라던지.. 이미 인간 실격 , 구의 증명은 다 봤습니다!! 근데 오늘밤~뭐시기나 위험한 편의점? 같은 일본 로맨스? 책은 추천하지말아주세요ㅠㅠ",
    "completion": "슬픈 소설을 추천해 드리겠습니다:\n\n"
}

{
    "prompt": "book_recommendation: {input에 주어진 책 정보와 사용자 질의를 바탕으로 그 책을 추천한 사유를 생성해줘}",
    "input": "user_query: {제가 한 달 뒤에 읽은 책 발표를 하는데 무슨 책을 읽어야할지 모르겠어요 중학생이 읽을만한 책 추천 해주세요 ㅠ}, book: {title: [청소년을 위한 진로상담], author: [정영선], introduction: [청소년을 위한 진로상담에 관한 개론서. 중학생진로와부모, 진로상담사례, 창의적책읽기를통한자기주도학습및진로탐색, 중학생을위한진로프로그램 등의 내용으로 구성했다.]}",
    "completion": "- 이 책은 중학생들을 위한 진로 상담에 대한 개론서로, 중학생들의 진로에 대한 이해를 돕고 자기주도학습 및 진로탐색에 대한 정보를 제공합니다."
}
```

Completion 부분은 '이 책은 ~' 으로 시작하게끔 학습 데이터가 구성되어 있다.

**※ 주의시항**

데이터의 형식을 정확히 지켜줘야 한다. 특히 서로 다른 괄호들이 서로 섞여 있는데 이를 혼동하면 학습이 잘 안될 수 있다.

## Loading Model and Data
통합 모델 코드가 있는 `consolidated_model_train.py`를 기준으로 작성했습니다. 다른 파일들도 개별 모델 학습이라 사실상 동일합니다.
```
MODEL_ID = "rycont/kakaobrain__kogpt-6b-8bit"
```
MODEL_ID 부분을 변경하면 다른 모델로도 학습이 가능하다. 생성형 모델 (`GPT`, `Polyglot` 계열)의 모델들을 사용 가능하다. 실제로 가능한지는 각 모델의 specification 참고

**참고 방법**

원하는 모델을 찾은 뒤, `Files and Versions`에 들어가면 모델마다 `README` 또는 `config.json`이 있을 것이다. 파일에 들어가서 해당 모델이 `CausalLM`인지를 확인. 맞다면 transformers 버전이 해당 모델을 지원하지 않는 것이 아닌 이상 해당 코드로 학습이 가능할 것이다.

`HuggingFace`에서 모델 및 tokenizer 불러오기
```
# get model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,
                                         padding_side="right",
                                         model_max_length=768)  # input의 최대 길이는 768
tokenizer.add_special_tokens(
    {
        "eos_token": EOS_TOKEN,
        "bos_token": BOS_TOKEN,
        "unk_token": UNK_TOKEN,
    }
)
tokenizer.pad_token = tokenizer.eos_token
# custom token들 추가하는 과정 
tokenizer.add_tokens([CLUE_TOKEN, REASONING_TOKEN, LABEL_TOKEN], special_tokens=True)
model.resize_token_embeddings(len(tokenizer)) # token 이 추가되었으니, model의 embedding 크기를 다시 맞춰춰야 한다.
```

`model_max_length`를 조절해서 모델이 받을 수 있는 입력의 최대 길이를 정해줄 수 있다.

데이터 로드
```
# load data for each task
intention_dataset = SFT_dataset(data_path_1_SFT="/path/to/intention/data.json", tokenizer=tokenizer, task="intention", verbose=True)
evaluation_dataset = SFT_dataset(data_path_1_SFT="/path/to/evaluation/data.json", tokenizer=tokenizer, task="evaluation", verbose=True)
generation_dataset = SFT_dataset(data_path_1_SFT="/path/to/recommendation/data.json", tokenizer=tokenizer, task="generation", verbose=True)

# concatenate all dataset
train_dataset = ConcatDataset([intention_dataset, evaluation_dataset, generation_dataset])
evaluation_dataset = None  # 통합 모델에 evaluation dataset은 사용하지 않음
data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)  # text padding을 학습하는 중에 실행해주는 data collator
```

각 데이터 별로 JSON 파일에 저장해서 올리면 된다. `generation_dataset`의 경우, 멘트 생성과 추천 사유 생성 관련 데이터가 함께 들어있기 때문에 실제 데이터를 processing하는 과정의 코드가 다음과 같다
```
        # class SFT_Dataset 에서 __init__ 메소드의 일부

        sources = []
        targets = []

        # inputs: 모델이 생성을 위해 받는 입력. 사용자가 chatbot에 질의한 문장이 들어오게 된다
        # targets: 모델이 생성해야 하는 것
        if task == "intention":  # intention 데이터
            prompt_input = PROMPT_DICT["intention"]
            for example in list_data_dict:
                sources.append(prompt_input.format_map({"input": example["input"]}))
                targets.append(f"{CLUE_TOKEN}단서들: {example['clues']}{CLUE_TOKEN}\n{REASONING_TOKEN}추론: {example['reasoning']}{REASONING_TOKEN}\n{LABEL_TOKEN}의도: {example['intention']}{LABEL_TOKEN}{tokenizer.eos_token}")
        elif task == "evaluation": # evaluation 데이터
            prompt_input = PROMPT_DICT["evaluation"]
            for example in list_data_dict:
                sources.append(prompt_input.format_map({"input": example["input"]}))
                targets.append(f"{example['completion']}{tokenizer.eos_token}")
        else:  # generation 데이터
            for example in list_data_dict:
                if example["prompt"].startswith("response"):  # 소개 관련 데이터
                    prompt_input = PROMPT_DICT["introduction"]
                else:  # 추천 사유 관련 데이터
                    prompt_input = PROMPT_DICT["generation"]
                sources.append(prompt_input.format_map({"input": example["input"]}))
                targets.append(f"{example['completion']}{tokenizer.eos_token}")
```

멘트 관련 데이터와 추천 사유 관련 데이터를 분리해서 보관한다면 코드를 if문의 task를 추가해서 코드를 다르게 작성할 수 있을 것이다.

최종 결과물은 아래 예시와 같은 데이터들이 tokenize 된 `torch.utils.data.Dataset` instance이다.
```
이 모델은 '입력 문장'을 분석해서 의도를 분류하는 모델입니다. 분류할 의도로는 [메타 정보 검색], [키워드 검색], [유사도서 검색], [판매량 검색], 그리고 [그 외]가 있습니다. 입력 문장이 책 추천과 관련이 없는 경우 [그 외]로 분류합니다. 입력 문장이 책 추천과 관련이 있는 경우, 다음과 같이 분류합니다. 저자, 출판사, 기간 등 메타 정보가 단서에 포함된 경우 [메타 정보 검색]으로 분류합니다. 키워드 정보만 단서에 포함된 경우 [키워드 검색]으로 분류합니다. 제목 정보가 단서에 포함된 경우 예외 없이 [유사도서 검색]으로 분류합니다. 판매량 정보가 단서에 들어간 경우 예외 없이 [판매량 검색]으로 분류합니다. 우선, 입력 데이터를 받으면 주어진 입력 데이터에서 의도를 분류할 때 도움이 될 수 있는 단서들을 추출합니다. 그리고 각 단서마다 어떤 종류의 단서인지 표시합니다. 그 다음, 주어진 입력 데이터와 단서들을 바탕으로 어떤 의도인지 추론하는 글을 생성합니다. 그 후, 입력 데이터와 단서들과 추론한 글을 바탕으로 주어진 5가지 의도 중 하나를 생성합니다. [의도: ]를 적고 생성합니다. 반드시 [단서들], [추론], [의도] 순서대로 생성해야 합니다. 의도를 생성할 때 반드시 주어진 글자들과 동일하게 생성해야 합니다.
입력: 안재성 작가가 쓴 책을 추천해줘


<CLUE>단서들: [안재성:저자]<CLUE>
<REASONING>추론: 입력 문장은 책 추천을 원하고 있습니다. 이는 '책을 추천해줘'를 통해서 확인할 수 있습니다. 단서를 보면, 저자가 '안재성'인 책을 원하고 있습니다. 따라서 입력 문장의 의도는 '메타 정보 검색'입니다.<REASONING>
<LABEL>의도: 메타 정보 검색<LABEL></s>
```
```
EVALUATION: 주어진 사용자의 질문과 정보를 바탕으로 한 책 추천의 적절성을 평가해 주세요. 만약 추천된 책이 사용자의 QUERY과 관련되어 있고 INFO에 잘 부합한다면 'Pass'을, 그렇지 않다면 'Fail'을 부여하세요. 책의 주제, 사용자 질문과의 관련성, 그리고 사용자의 필요에 대한 적합성을 고려하세요. 추천이 적절하고 관련이 있는지 평가하는 것이 중요합니다.
QUERY : {다문화와 관련된 책 추천해줘}, INFO : {title='한국 사회의 다중언어 의사소통' introduction='한국은 다문화, 다중언어사회로 빠르게 진입하고 있다. 2020년 기준 국내 체류 외국인 수는 200만 명 이상이며, 이들 한국에 거주하는 외국인 유학생, 근로자, 결혼이민자들은 한국인과의 접촉에서 혹은 한국에 살고 있는 다른 언어권 사람들과의 의사소통에서 한국어를 제1언어로 사용하는 한편, 국제어인 영어를 비롯한 다양한 언어적 자원을 활용하여 의사소통한다. 다중언어 사회의 언어 능력은 발음ㆍ어휘ㆍ문법의 정확성만으로는 측정할 수 없으며, 다양한 전략과 방법들이 의사소통에을 성공적으로 이끌기 위해 동원된다. 이 책은 이러한 횡단적 언어 수행의 관점에서 한국 사회의 다중언어 의사소통 양상을 고찰한다. 한국인을 포함하여 다양한 국적의 다양한 언어를 사용하는 이들이 한국이라는 공간에서 의사소통을 위해 어떠한 전략과 방법을 사용하는지, 대화를 어떠한 구조로 전개하는지, 한국의 공간들은 어떠한 다중언어 경관을 보이는지, 외국인의 발화에 대한 한국인의 태도는 어떠하며, 한국어가 모어(母語)가 아닌 이들의 한국어 사용을 우리 사회는 어떠한 시선으로 바라보는지 등의 문제를 다룬다. 더불어 다중언어주의의 관점에서 한국어교육의 목표는 ‘정확한 언어 지식의 습득‘보다는 ‘유창한 언어 사용 능력의 향상’을 중시하는 방향으로 전환되어야 한다고 제안한다.' author='이정란, 지현숙, 백승주, 황선영' publisher='한국학중앙연구원출판부' isbn=9791158666835}


Pass</s>
```
```
### Prompt(명령):
book_recommendation: {input에 주어진 책 정보와 사용자 질의를 바탕으로 그 책을 추천한 사유를 생성해줘}

### Input(입력):user_query: {제가 한 달 뒤에 읽은 책 발표를 하는데 무슨 책을 읽어야할지 모르겠어요 중학생이 읽을만한 책 추천 해주세요 ㅠ}, book: {title: [청소년을 위한 진로상담], author: [정영선], introduction: [청소년을 위한 진로상담에 관한 개론서. 중학생진로와부모, 진로상담사례, 창의적책읽기를통한자기주도학습및진로탐색, 중학생을위한진로프로그램 등의 내용으로 구성했다.]}

### Response(응답):
- 이 책은 중학생들을 위한 진로 상담에 대한 개론서로, 중학생들의 진로에 대한 이해를 돕고 자기주도학습 및 진로탐색에 대한 정보를 제공합니다.</s>
```

evaluation data를 사용할 경우, training_data에서 일부를 때어서 사용하면 된다. 다만 이번 통합 모델의 경우, 데이터의 양이 충분히 많지 않아서 전부 학습에 사용하기로 했다. Evaluation data를 만들고 싶다면 `torch.utils.data.random_split`을 활용하면 되고, [여기](https://076923.github.io/posts/Python-pytorch-11/)를 참고하면 될 것이다.

## LoRA Configuration
`peft` 라이브러리를 사용
```
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)
```

설정 가능한 값:
|하이퍼파라미터|설명|값|
|---|---|---|
|r|Low intrinsic Rank, 원본 모델을 근사한 rank값|8|
|lora_alpha|학습 시 LoRA가 영향을 끼치는 정도. 여러 논문에서 16을 고정으로 사용|16|
|lora_dropout|과적합 방지를 위해 일정 cell들을 학습 중에 비활성화하는 비율|0.05|

원본 모델에 LoRA 설정을 추가
```
model = get_peft_model(model, config)
```
LoRA 설정된 모델을 학습 시, 원본 모델의 weight 값이 바뀌는 것이 아닌 adapter 파일이 따로 생성됨.

## Training Arguments
`TrainingArguments`를 사용해서 학습 시 사용할 하이퍼파라미터들을 설정

```
training_args = TrainingArguments(
    output_dir="./consolidated_models/kakao_models/data_300_n_100_lr_3e_5", # 학습된 모델 저장 경로
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=100, # number of training epochs
    per_device_train_batch_size=1, # batch size for training. GPU 크기 문제로 batch size는 1로 해야 했다
    per_device_eval_batch_size=1,  # batch size for evaluation
    eval_steps = 5, # Number of update steps between two evaluations.
    save_steps=10, # after # steps model is saved
    #evaluation_strategy="steps",
    logging_steps=10,
    warmup_steps=5,# number of warmup steps for learning rate scheduler
    prediction_loss_only=True,
    fp16=True,  # 공간 절약을 위해 floating point 16d으로 학습을 진행 (물론 현재 사용하는 모델이 fp16이라 큰 상관은 없다)
    gradient_accumulation_steps=16,  # GPU 공간 절약을 위해 추가한 hyperparameter
    learning_rate=3e-5,  # learning rate은 조금 작게 하고 오랫동안 학습시키는 방식을 택함 (1e-05 3e-05 테스트 해봄)
    lr_scheduler_type="cosine",  # learning rate가 시간이 갈수록 점점 작아지게끔 하는 scheduler 사용
    #load_best_model_at_end=True, # always save the best model
    #save_total_limit=10  # total saved checkpoints are 10
)
```
|파라미터|설명|값|
|---|---|---|
|output_dir|학습된 모델 저장 경로.||
|overwrite_output_dir|새로 학습할 시 output_dir에 명시된 곳에 이미 모델이 있을 경우 덮어씌운다. **실수로 덮어씌우는 실수를 하지 않게 조심해야 한다.**|True|
|num_train_epochs|전체 데이터를 학습하는 횟수. 100을 넘어가면서부터는 모델의 성능이 변한다는 느낌이 없음|100|
|per_device_batch_size|학습 데이터 batch size. GPU의 메모리 용량에 맞춰서 설정해주면 된다|1|
|gradient_accumulation_steps|gradient descent를 보류할 step의 수. 매 step마다 학습하는 것이 아닌 일정 step마다 하는 것이기에 메모리 절약에 도움된다. 가능하면 줄이는 것이 좋다|16|
|save_steps|checkpoint를 저장하는 step interval. 10으로 설정하면 10step마다 model의 checkpoint가 생성되고, 각 모델을 사용할 수 있다.|10|
|logging_steps|loss 와 learning rate 등 step 마다 변동하는 값은 기록하는 interval|10|
|prediction_loss_only|prediction loss만 기록|True|
|fp16|16bit 실수형 자료구조로 학습 진행 여부. 모델이 이미 양자화 되어있는 경우, 무시된다.|True|
|learning_rate|학습률|3e-5|
|lr_scheduler_type|학습을 하면서 점차 learning rate를 감소시키는 방식. Cosine이 가장 무난한 선택지로 보인다|"cosine"|
|warmup_step|learning rate scheduler를 적용하기 전 기다리는 step의 수|5|

그 외에 evaluation 데이터가 있다면 다음 파라미터 사용을 고려해볼 수 있다
|파라미터|설명|
|---|---|
|eval_steps|evaluation이 진행되는 step interval|
|per_device_eval_batch_size|한번에 evaluation을 진행할 batch size|
|load_best_model_at_end|True로 설정하면 가장 좋은 모델을 저장한다. 좋은 모델이라는 것은 training loss가 낮으면서 validation loss도 낮은 상태를 의미한다|
|save_total_limit|너무 많은 checkpoint들이 저장되는 것을 방지하기 위해 마지막 몇 개의 checkpoint만 저장할 수 있다. load_best_model_at_end 까지 설정되어 있으면 best model도 포함해서 저장된다|

## Train
```
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=evaluation_dataset
)
```
Trainer에 학습할 `model`과 `training_args`, `data_collator`, `train_dataset`, `eval_dataset`을 넣어서 trainer instance를 만든다.

`trainer.train()` 을 실행하면 학습이 된다.

## Background 학습 방법

터미널에 다음과 같이 입력한다.
```
nohup python consolidated_model_train.py > output.log &
```
모든 결과물을 `output.log`에 redirect 하고 `nohup`을 사용하면 시스템의 전원이 off돼도 학습이 멈추지 않는다. 도중에 진행 경과를 확인하기 위해서는 
```
cat output.log
```
를 해주면 된다.

## Generation
`./generating_with_model` 디렉토리에 위치
- `intention_generation.py` : 의도 분류 생성
- `evaluation_generation.py` : 추천 도서 평가 생성
- `book_recommendation_generation.py` : 멘트 및 추천 사유 생성
- `consolidated_generation.py` : 통합 모델 생성

통합 모델 코드 기준으로 설명한다. 다른 코드들은 개별 코드들이고 사실상 동일하다.

## Preparation
학습을 했을 때와 동일한 설정으로 model과 tokenizer를 불러와야 한다.
```
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
```
이 때 불러오는 모델은 `Huggingface`에서 가져온 fine-tune이 되지 않은 **원본 모델**이다. LoRA를 통해 학습된 것은 모델이 아니라 모델을 감싼 adapter이다.

`peft`를 활용해서 학습 과정에서 저장된 checkpoint를 가져와서 원본 모델에 부착한다. 아래는 예시이다.
```
model = PeftModel.from_pretrained(
    model=model,
    model_id="./consolidated_models/kakao_models/n_100_lr_3e_5/checkpoint-9300/",
)
```
학습된 adapter의 위치는 학습할 때 명시한 `output_dir`안에 있는 여러 `checkpoint`파일 안에 있다. [여기](https://huggingface.co/docs/transformers/main/peft)를 참고

**중요**

정상적으로 학습되었다면 각 `checkpoint`파일 안에 `adapter_config.json`이라는 파일이 들어있을 것이다. 만약 들어있지 않다면 위 코드를 실행하면 `could not find adapter_config.json` 에러가 뜨면서 실행에 실패할 것이다. 이 경우에는 학습에 실패한 것이다. 학습을 했는데도 불구하고 `adapter_config.json`이 정상적으로 생성되지 않는 원인을 정확하게 파악하지는 못했지만, 주요 원인은 패키지 간의 호환성 문제가 대다수였다.

## Generation Arguments
Task별 생성 함수에는 `generation_args`에 생성 관련된 하이퍼파라미터들이 있다.

```
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
```

|하이퍼파라미터|설명|값|
|---|---|---|
|num_beams|beam search의 beam 개수, 모델이 다음에 올 토큰을 예측할 때 한가지만 선택하는 것이 아닌 여러 개의 beam을 선택해서 그 중 가장 괜찮은 거를 고르는 방식이다|2|
|repetition_penalty|다음 토큰이 이전 토큰과 동일한 경우, 해당 토큰에 어느정도 패널티를 부과할지애 관한 파라미터|2.0|
|no_repeat_ngram_size|n개의 연속된 token이 여러번 반복되는 걸 방지|4|
|max_new_tokens|최대 생성 가능 토큰 수|512|
|do_sample|beam search를 위해 설정|True|
|top_p|다음 토큰을 예측할 때 상위 p%의 토큰들만을 고려하게끔 해서 생성 결과가 다양해지게 하는 것을 방지 (0과 1 사이의 값만 가능)|0.15|
|early_stopping|beam search가 끝남과 동시에 생성을 종료할지 여부|True|
|temperature|작을 수록 생성 결과가 똑같이 나오고 클수록 다양해짐. 다만 이 값을 1.0 미만으로 줄일 시 생성 과정에서 확률값이 표현 가능한 숫자 범위를 벗어나는 에러가 발생해서 생성이 실패하는 경우가 잦아서 기본값을 사용|1.0|

## Generation
각 함수가 거의 동일하게 작동한다.
```
def determine_intention(user_input: str, model=model, tokenizer=tokenizer):
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    generation_args = dict(
        num_beams=2,  # test  2, 3
        repetition_penalty=2.0,
        no_repeat_ngram_size=4,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.15,
        early_stopping=True,
    )

    mapped_prompt = PROMPT_DICT["intention"].format_map({"input": user_input})
    response = generator(mapped_prompt, **generation_args)  # generate
    result = (response[0]["generated_text"]).replace(
        mapped_prompt, ""
    )  # response to query
    return result
```
```
def generate_evaluation(user_input: str, model=model, tokenizer=tokenizer):
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

    mapped_prompt = PROMPT_DICT["evaluation"].format_map({"input": user_input})
    response = generator(mapped_prompt, **generation_args)
    result = (response[0]["generated_text"]).replace(mapped_prompt, "")
    return result
```
```
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
    result = translate_text(target_lang, result)
    outstring = result + "<br>"
    # book dataset of book data
    list_book = [
        PROMPT_DICT["generation"].format_map(
            {"input": "user_query: {" + user_input + "}, " + book}
        )
        for book in book_data
    ]
    list_result = generator(list_book, **generation_args)

    pattern = r"title\s*=\s*\[([^\]]+)\],\s*author\s*=\s*\[([^\]]+)\]"

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
            + str(isbn)
            + '" target="_blank" class="quickViewButton">Quick View</a><br><br>'
        )
    return outstring
```
들어온 데이터에 task에 맞는 prompt를 붙이고 `pipeline`을 사용해서 텍스트 생성기를 만든 후, 생성을 해주면 된다. 그 뒤, prompt와 input 부분만 없애주면 생성이 완료된다.

###각 task 별 예시 input & output

```
# intention generation
result = determine_intention(model, 
                             tokenizer, 
                             user_input="TOPIK 공부할 수 있게 책 추천해줘")

# output
단서들: [TOPIK:키워드, 공부할 수있게:키워드]
추론: 입력 문장은 책 추천을 원하고 있습니다. 이는 '책 추천해줘'를 통해서 확인할 수 있습니다. 단서를 보면, 'TOPIK'를 키워드로 하는, '공부할 수있도록'을 키워드로 하면서 'TOPIK'인 책을 원하고있는 것을 알 수 있습니다 따라서 입력 문장의 의도는 '메타 정보 검색'입니다
의도: 메타 정보 검색
```

```
# evaluation generation
result = generate_evaluation(model, 
                    tokenizer, 
                    user_input="QUERY : {이스라엘과 팔레스타인 관련된 도서를 알려줘}, INFO : {title = '팔레스타인과 이스라엘의 분쟁' introduction = '이 책은 팔레스타인 지역을 둘러싼 팔레스타인 아랍인과 이스라엘 유대인의 분쟁의 역사를 담고 있다. 연이은 전쟁으로 수많은 사람들의 희생이 거듭되고 있음에도 분쟁 해결이 쉽지 않은 곳, 바로 팔레스타인의 역사인 셈이다. 팔레스타인 분쟁의 시작은 아주 오래 전 팔레스타인에 살았던 유대인들이 시오니즘 운동을 토대로 1948년 팔레스타인에 유대 민족 국가인 이스라엘을 건국하면서 비롯되었다. 이스라엘의 건국은 팔레스타인에 살고 있던 아랍인들에게는 조상 대대로 살아온 삶의 터전을 잃게 됨을 의미했다. 또한 이스라엘 건국은 거대한 문명 충돌을 가지고 왔다. 유대 민족의 옛 수도 예루살렘은 유대 민족이 유일신으로 섬기는 여호와의 성전이 있던 곳이자, 예수 그리스도가 십자가에 못 박혀 죽은 곳이며 동시에 이슬람교 창시자 무함마드가 하늘로 승천한 이슬람교 3대 성지 중 하나였다. 결국 한 지역에 유대교, 기독교, 이슬람교가 공존하게 되었고, 이후 예루살렘은 그 누구도 양보할 수 없는 분쟁의 중심에 서게 되었다. 이 책은 이처럼 팔레스타인 분쟁의 시발점이 되었던 이스라엘 건국을 기점으로 이후 팔레스타인 아랍인과 이스라엘 간에 일어났던 크고 작은 사건들을 상세히 설명한다. 덕분에 네 차례에 걸쳐 일어난 중동전쟁은 물론 냉전시대 이후 본격화되었던 중동 평화를 위한 노력이 어떻게 전개되었는지 또한 쉽게 이해할 수 있다. 뿐만 아니라 유대인들의 미국 내 영향력, 팔레스타인 아랍인들의 독립을 위한 노력, 이스라엘을 지원하는 미국을 향한 이슬람원리주의자들의 과격한 테러 등 팔레스타인 지역을 둘러싼 사건과 인물에 대한 친절한 설명은 팔레스타인과 이스라엘 분쟁에 대한 이해를 높인다. 특히 이 책에서 주목하고 있는 것은 팔레스타인과 이스라엘 분쟁으로 인해 생겨난 여러 가지 문제들이다. 삶의 터전을 빼앗긴 지 60여 년이 지난 지금도 집으로 돌아갈 수 없는 팔레스타인 난민들, 이들을 수용하기에는 턱없이 부족한 정착지, 부족한 수자원 등을 해결하기 위해서 국제 사회는 어떤 역할을 해야 하는지에 대한 깊은 고민이 담겨 있다.' author='손기화' publisher='주니어김영사' isbn=9788934955788}")

# output
Pass
```

```
generate_recommendation(model=model, 
                     tokenizer=tokenizer, 
                      user_input='사회비판의 내용을 담은 시집 추천', 
                      book_data=['book: {title: [한국리얼리즘 한시의 이해], author: [정양 외], introduction: [현직교수가 한국 한시의 현실인식에 대해 작품예와 함께 탐구한 저서. 현실주의 한시의 특질,  현실주의 한시의 역사적 흐름,봉건체제의 모순과 현실비판, 피폐된 농촌 현실과 삶의 갈등, 건강한 전가생활과 공동체적 삶의 추구 등으로 엮었다.]}'])

#output
사회비판적인 내용을 담고 있는 시집을 추천해 드리겠습니다:


[한국리얼리즘 한시의 이해] (정양 외 저)
- 이 책은 현직 교수가 한국 한시에 대한 현실인식을 작품예와 더불어 탐구한 연구서로, 현실주의 한시의 특징, 현실주의 한시가 역사적으로 어떻게 변화해왔는지, 봉건체제의 모순 및 현실비판, 농촌의 현실과 삶과의 갈등, 건강하고 지속가능한 공동체적 삶 등의 주제를 다루고 있습니다.
```
---

작성자 : 권규영
