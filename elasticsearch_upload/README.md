# Elasticsearch 데이터 업로더

이 스크립트는 CSV 파일에서 Elasticsearch 인덱스로 데이터를 업로드하기 위해 설계되었습니다. `sentence-transformers`, `elasticsearch`, `pandas`, `tqdm` 파이썬 패키지를 사용하여 데이터를 효율적으로 처리하고 업로드합니다.

## 사용한 라이브러리

- `sentence_transformers`: 문장 임베딩 생성을 위해 사용
- `pandas`: CSV 파일 처리 및 데이터 조작을 위해 사용
- `elasticsearch`: Elasticsearch와의 연결 및 데이터 업로드를 위해 사용
- `tqdm`: 데이터 처리 진행 상황을 시각적으로 표시하기 위해 사용

## Installation
```
pip install sentence_transformers
pip install pandas
pip install elasticsearch
pip install tqdm
```
- elasticsearch 홈 디렉토리에서
```
$ bin/elasticsearch-plugin install analysis-nori
```

## 데이터 CSV 형식

- CSV 파일은 UTF-8 인코딩을 사용해야 합니다.
- 다음과 같은 행이 포함되어야 합니다: `author`, `category`, `introduction`, `publisher`, `title`, `publish_date`, `isbn`, `toc`.
- 각 행은 적절한 데이터 타입을 가져야 합니다 (예: `publish_date`는 날짜 형식).
- CSV 파일은 전처리가 진행된 상태여야 합니다.
  - **각 book information에 대하여 publish_date 항목은 반드시 date 형식이어야 합니다.**
  - 나머지 항목에 대해서는 빈칸이 되지 않도록 처리해야합니다.
  - html태그가 존재할 경우, 없애는 것을 추천합니다.
  
| `author` | `category` | `introduction` | `publisher` | `title` | `publish_date` | `isbn` | `toc` |
|----------|------------|----------------|-------------|---------|----------------|--------|-------|
| 책의 저자 이름. | 책의 카테고리 또는 장르. | 책의 간략한 소개 또는 요약. | 책을 출판한 출판사 이름. | 책의 제목. | 책의 출판 날짜. 일반적으로 'YYYY-MM-DD' 형식을 사용합니다. | 책의 국제 표준 도서 번호(ISBN). | 책의 목차(Table of Contents). |

## 설정

1. **Elasticsearch 연결 설정**:

   ```python
   es = Elasticsearch(
       ["https://your_elasticsearch_server:port"],
       basic_auth=("username", "password"),
       verify_certs=False,
       timeout=30,
       max_retries=10,
       retry_on_timeout=True,
   )
   ```
- default
  - `["https://your_elasticsearch_server:port"]`: Elasticsearch 서버의 URL과 포트 번호를 목록 형태로 지정합니다.
- if needed
  - `basic_auth=("username", "password")`: Elasticsearch 서버에 접근하기 위한 기본 인증 정보입니다. 
  - `verify_certs=False`: SSL 인증서 검증을 비활성화합니다. 보안이 중요한 환경에서는 `True`로 설정하는 것이 좋습니다.
  - `timeout=30`: 클라이언트의 요청 타임아웃 시간(초)을 설정합니다.
  - `max_retries=10`: 최대 재시도 횟수를 지정합니다. 연결 실패 또는 타임아웃 발생 시 클라이언트가 재시도하는 횟수입니다.
  - `retry_on_timeout=True`: 타임아웃 발생 시 재시도할지 여부를 결정합니다. `True`로 설정하면 타임아웃 발생 시 재시도를 시도합니다.
  
2. **업로드 설정**:
    ```
    model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    input_filename = f"csv file directory to upload"
    chunksize = 50
    index_name = "data"
    ```
    - `model` : embedding을 할 때 사용할 모델 지정
    - `input_filename` : 도서 데이터가 담긴 csv 파일의 경로
    - `chunksize` : upload 하는 단위인 chunk의 크기 지정
    - `index_name` : 도서 데이터를 upload 할 Elasticsearch의 index name

3. **인덱스 설정 (`setting`과 `mapping`)**:
   
   - `setting`: 분석기와 토크나이저를 정의합니다. 예를 들어, 한글 텍스트 분석을 위한 Nori 분석기 설정은 다음과 같습니다:

     ```python
     setting = {
         "analysis": {
             "analyzer": {
                 "nori_analyzer": {
                     "type": "nori",
                     "tokenizer": "nori_mixed"
                 }
             },
             "tokenizer": {
                 "nori_mixed": {
                     "type": "nori_tokenizer",
                     "decompound_mode": "mixed",
                 }
             }
         }
     }
     ```

     **setting**:
     - analyer와 tokenizer로 노리(nori) 한글 형태소 분석기를 사용합니다.
     - `decompound_mode`: 옵션을 통해 합성어의 저장 방식을 결정합니다.
       - `none`: 어근을 분리하지 않고 완성된 합성어만 저장합니다.
       - `discard`: 합성어를 분리하여 각 어근만 저장합니다.
       - `mixed`: 어근과 합성어를 모두 저장합니다.

   - `mapping`: Elasticsearch 인덱스에 저장될 각 필드의 타입과 속성을 정의합니다. 


     ```python
     mapping = {
         "properties": {
             "author": {"type": "text", "analyzer": "nori_analyzer"},
             "category": {"type": "text", "analyzer": "nori_analyzer"},
             "introduction": {"type": "text", "analyzer": "nori_analyzer"},
             "publisher": {"type": "text", "analyzer": "nori_analyzer"},
             "title": {"type": "text", "analyzer": "nori_analyzer"},
             "publish_date": {"type": "date"},
             "isbn": {"type": "unsigned_long"},
             "toc": {"type": "text", "analyzer": "nori_analyzer"},
             "embedding": {
                 "type": "dense_vector",
                 "dims": 768,
                 "index": True,
                 "similarity": "cosine"
             }
         }
     }
     ```
     **mapping**:
     - elasticsearch에 올릴 항목들의 properties를 지정
     - `author`, `category`, `introduction`, `publisher`, `title`, `toc` 
       - field type : `text` 
       - analyzer : `nori_analyzer`
     - `publish_date`
       - field type : `date`
     - `isbn`
       - field type : `long`
     - `embedding` : KNN 기반 유사도 검색에 대응
       - field type : `dense_vector`
       - dims : 벡터 차원 수 지정
       - index : 벡터 필드 색인 유무
         - true여야 검색 쿼리에서 벡터 유사성을 검색 가능
       - similarity : 벡터간 유사성 측정 방법
  
4. **데이터 처리**
    ```
    data = []

    def appendbulk(row):
        targetstring = f"category: {row['category']}, author: {row['author']}, introduction: {row['introduction']}, title: {row['title']}"
        embedding = model.encode(targetstring)

        data.append({
            "_index": "data",
            "_source": {
                "author": row["author"],
                "category": row["category"],
                "introduction": row["introduction"],
                "publisher": row["publisher"],
                "title": row["title"],
                "publish_date": row["publish_date"],
                "isbn": row["isbn"],
                "toc": row["toc"],
                "embedding": embedding,
            },
        })
    ```
    - embedding 진행
      - 대상 : `targetstring`
        - `category`, `author`, `introduction`, `title`를 한 문장으로 연결
    - data append
      - `_index`: 업로드할 Elasticsearch index name
      - `_source`: Elasticsearch에 upload할 항목 지정

5. **데이터 업로드**
   
   - CSV 파일을 읽고, 청크 단위로 데이터를 처리한 후 Elasticsearch 인덱스에 업로드합니다.
    ```
    with pd.read_csv(input_filename, chunksize=chunksize, encoding="utf-8") as reader:
    for chunk in tqdm(reader):
        print("--------------------------------------")
        chunk.apply(appendbulk, axis=1)
        bulk(es, data)
        data.clear()
    ```   
    
## 사용법

-  Python 환경에서 스크립트를 실행합니다.
   ```bash
   python /Book-rec-with-LLM-refactored/elasticsearch_upload/sentence_transformer_encoding.py
   ```

## 주의 사항

- Elasticsearch 인스턴스의 설정과 연결을 확인하세요.
- CSV 파일 형식과 data type이 올바른지 확인하세요. 만약 맞지 않는 경우 upload가 진행되지 않습니다.
---
작성자 :  박건우