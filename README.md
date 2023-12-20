# Book rec with LLM
book recommendation system for KnP
## 폴더 구성
```
│  .gitattributes
│  .gitignore
│  config.json //실행사항의 config
│  keys.py.example //keys.py 으로 사용해서 필요한 key를 입력
│  README.md
│
├─elasticsearch_upload //엘라스틱서치 document업로드
│      README.md
│      sentence_transformer_encoding.py
│
├─elasticsearch_util 
│  │  elasticsearch_retriever.py //엘라스틱서치 검색용 function 구현
│  │  __init__.py
│
├─LLMs //GPT API 및 gpt기반 opensourceLLM의 학습 및 생성
│  │  README.md
│  │  __init__.py
│  │
│  ├─actual_models
│  ├─generating_with_model //모델을 통한 생성. 각 코드 실행시 생성 endpoint 시작
│  │      book_recommendation_generate.py //추천사유 생성
│  │      consolidated_generation.py //통합모델이 생성. 엔드포인트에 따라 추천사유, 평가, 의도추출을 진행
│  │      evaluation_generation.py //평가모델 생성
│  │      intention_generation.py //의도추출모델 생성
│  │      __init__.py
│  │
│  ├─GPT_API_utils //GPT API를 사용한 생성.
│  │  │  Generate_Recommendation.py //추천사유 생성 함수 포함
│  │  │  Recommendation_Evaluation.py //추천평가 생성 포함
│  │  │  __init__.py
│  │ 
│  ├─training_model //각 모델의 학습코드
│  │      book_evaluation_generation_train.py
│  │      book_recommendation_train.py
│  │      consolidated_model_train.py
│  │      intention_train.py
│
├─SBERT //도서 임베딩을 위한 SBERT의 학습 및 생성
│      README.md
│      unsup_CL_generation.py //SBERT 생성
│      unsup_CL_train.py //SBERT 학습
│
├─system_manager //백엔드 시스템
│  │  without_langchain.py //langchain을 사용하지 않는 flow. multi-turn(memory) 불가
│  │  __init__.py
│  │  
│  ├─with_langchain //langchain을 사용하는 flow. multi-turn(memory) 가능
│  │  │  with_langchain.py
│  │  │  __init__.py
│  │  │
│  │  └─modifiedLangchainClass //수정한 langchain코드
│  │          elastic_search_bm25.py //현재는 사용하지 않고 위 elasticsearch_util.elasticsearch_retriever 사용.
│  │          openai.py //agent에 대한 코드. 강제 종료 및 생성의 failmode에 대한 handling
│  │          schema.py 
│
├─utils //파파고를 이용한 번역
│  │  translation.py 
│  │  __init__.py
│
├─web //웹 구현
│  │  process.py //실행시 웹앱 엔드포인트 생성 /demo로 접속
│  │
│  ├─static
│  │      demo_mobile.css
│  │
│  └─templates
│          index_mobile.html
│          __init__.py
```
## 실행 방법
### 파이썬 라이브러리
파이썬 3.8.13  
Book-rec-with-LLM-refactored/ 디렉토리에서 작업  
pip install -r requirements.txt
### 웹앱
Book-rec-with-LLM-refactored/ 디렉토리에서 작업  
./web/process.py 실행  
python ./web/process.py  
80번 포트에 웹서버가 오픈되고 /demo 주소로 웹페이지로 이동 가능
### 모델 endpoint
#### 통합모델
Book-rec-with-LLM-refactored/ 디렉토리에서 작업
./LLMs/generating_with_model/consolidated_generation.py 에 
```
model = PeftModel.from_pretrained(
    model=model,
    model_id="./consolidated_models/kakao_models/n_100_lr_3e_5/checkpoint-9300/",
)
```
model_id 의 path를 실제 모델이 저장되어 있는 path를 현재 디렉토리 Book-rec-with-LLM-refactored/ 의 상대주소로 설정  
./LLMs/generating_with_model/consolidated_generation.py 실행  
python ./LLMs/generating_with_model/consolidated_generation.py  
5001번 포트에 엔드포인트가 오픈.  
/generation/intention, /generation/evaluation, /generation/final 3개의 post request가 가능
#### 통합모델이 아닌 경우
마찬가지로 Book-rec-with-LLM-refactored/ 디렉토리에서 작업  
LLMs/generating_with_model/ 의 각 모델 py 파일에서 마찬가지로 상대경로로 path를 설정  
이후 각 모델 py파일 실행
500x번 포트에 엔드포인트 오픈  
/generation/&lt;task&gt; 로 post request가 가능
### elasticsearch
elasticsearch_upload README.md 참고해 업로드
업로드된 elasticsearch 실행 및 config에 주소 설정
## config
    "elasticsearch_result_count": 30, //엘라스틱서치에서 bm25+knn 검색할 도서 권수
    "default_number_of_books_to_return": 3, // 기본으로 검색할 최대 도서
    "elasticsearch_url": "elasticsearch_url", //엘라스틱서치 (보안 정보를 포함한) 주소 
    "elasticsearch_index_name": "data", //엘라스틱서치 인덱스 명
    "flowchoice": "langchain", //langchain 사용할지 여부 "langchain" 또는 "nolangchain"
    "mongodb_uri": "mongodb url", 
    "intention_generation_url": "http://127.0.0.1:5001/generate/intention", //의도파악 모델 엔드포인트
    "evaluation_generation_url": "http://127.0.0.1:5001/generate/evaluation", //평가 모델 엔드포인트
    "final_generation_url": "http://127.0.0.1:5001/generate/final", //최종추천사유 모델 엔드포인트
    "use_gpt_api_for_eval": false, //평가에 gpt api 사용 여부
    "use_gpt_api_for_final": false, //최종 추천사유 생성에 gpt api 사용여부
    "meta_search_canned_text_ko": "다음과 같은 도서가 검색되었습니다.", //메타정보 검색 출력
    "meta_search_canned_text_en": "Following is the search result.", //메타정보 검색 출력
    "no_book_found_text": "No books found. Please try again", //도서 검색 실패시 출력
    "filter_out_reccommended_books": true //langchain flow 에서 이미 추천된 도서 추천을 배제
