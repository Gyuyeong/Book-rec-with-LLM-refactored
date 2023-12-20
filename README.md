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
