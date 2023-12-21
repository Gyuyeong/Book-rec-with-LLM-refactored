# SBERT - Contrastive Learning
## 1. 생성 아웃풋
기존 KR-SBERT의 아웃풋과 같이 768차원의 embedding vector가 나오게 된다.

예시 :
```python
[ 1.73470461e+00  4.47719812e-01  2.20545799e-01 -3.92011315e-01
 -9.41857815e-01  1.12080884e+00  2.71359742e-01 -5.28857350e-01
                                .
                                .
                                .
 -7.11882889e-01 -4.46518838e-01 -9.76859272e-01 -5.73234200e-01]
```

어떤 문장을 embedding 하기 위해서 사용할 수 있는 함수이다.   
(Book-rec-with-LLM-refactored/SBERT/unsup_CL_generation.py에 정의)   
| variable | description |
|---|:---:|
| `sentence` | embedding 하고자 하는 문장 |
| `model_name` | 사용할 embedding model |

```python
def embedding_sentence(sentence, model_name):
    # CL 학습 완료한 SBERT 모델 로드
    model = SentenceTransformer(model_name)

    # 모델을 이용해 embedding 진행
    embedding = model.encode(sentence, convert_to_tensor=True).cpu().numpy()

    return embedding
```

## 2. 학습 데이터 생성
공앤박의 도서 데이터에서 category, author, title, introduction 열을 뽑아서, book_info열에 하나의 문장으로 합쳐서 넣어준다.

형식 :
```python    
category: {row['category']}, author: {row['author']}, title: {row['title']}, introduction: {row['introduction']}
```

예시 :
```python
category: 종교 > 기독교(개신교) > 기도/설교/전도 > 설교학, author: 곽선희, title: 참회의 은총(설교집 11), introduction: 이 책은 일반인들이 기독교에 대해 이해할 수 있는 교양서이다.
```

이때, Unsupervised Contrastive Learning의 효율을 높여주기 위해서는 하나의 batch 안에 들어갈 도서 데이터들이 서로에게 negative pair가 될 수 있도록 보장해줘야 한다.

이를 위해, 도서 데이터의 column 중 category에서 상위 두번째 카테고리까지를 기준으로 책 데이터를 분류하고, 이들 중 속하는 책의 수가 많으면서, 서로 최대한 겹치지 않는 내용을 담을만한 카테고리들을 batch size보다 많은 수만큼 뽑는다. 그리고 하나의 batch마다 모두 서로 다른 카테고리의 책들로만 구성될 수 있게 하면, 각 batch에 속하는 도서 데이터가 모두 서로 유사성이 낮은 데이터로 구성될 수 있으므로 이와 같이 학습 데이터를 구성했다.

## 3. 학습
### 3-1. Base 모델 로드
모델을 불러오기 위해 적절한 model name을 설정하고, 이를 통해 embedding model과 pooling model을 불러온다.
이 둘을 합쳐 학습할 Sentence Transformer 모델 로드를 완료한다.   
model_name을 바꾸어 Base 모델을 바꾸어 학습할 수 있다.
```python
model_name = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'
word_embedding_model = models.Transformer(model_name, max_seq_length=512)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
```

### 3-2. 학습 데이터 로드
Contrastive Learning을 진행하려면, 하나의 데이터에 대해 positive pair와 negative pair가 존재해야 한다. positive pair는 유사도가 더 높아지도록, negative pair는 유사도가 더 낮아지도록 학습이 진행된다.

그런데 Unsupervised Contrastive Learning에서는 positive pair를 별도의 문장으로 만드는 것이 아닌, 자기 자신의 문장을 embedding하면서 dropout을 적용하여 살짝 바뀐 embedding vector로 활용한다. 그래서 아래 코드와 같이 학습 데이터에 한 권의 책 데이터를 두 번 반복하여 넣어주고, 이를 통해 Contrastive Learning을 진행한다.
```python
data = pd.read_csv("unsupCLdata_shuffled_v3_212k.csv")

# 'book_info' 열에서 한 행씩 읽어서 train_samples(학습데이터) 리스트에 추가
# unsupervised Contrastive Learning을 위해 동일 데이터를 두 개씩 묶은 데이터가 들어가게 됨
train_samples = []
for index, row in data.iterrows():
    book_info = row['book_info']
    train_samples.append(InputExample(texts=[book_info, book_info]))
```

### 3-3. 학습 조건, 하이퍼파라미터
| parameter | description | value |
|---|:---:|---:|
| `max_seq_length` | embedding 시 max_seq_length 보다 긴 input은 truncate | 128 |
| `batch_size` | 하나의 batch에 얼마나 많은 수의 sample이 들어가는지 결정 | 64 |
| `epoch` | 전체 데이터셋을 몇 번 반복하여 학습할 것인지 결정 | 1 |
| `learning_rate` | 얼마나 빠른 속도로 학습을 진행할 것인지 결정 | 2e-5 |
```python
word_embedding_model = models.Transformer(model_name, max_seq_length=128)

train_batch_size = 64
train_dataloader = DataLoader(train_samples, shuffle=False, batch_size=train_batch_size)

num_epochs = 1
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          output_path=model_save_path,
          checkpoint_path=checkpoint_path,
          checkpoint_save_steps=len(train_dataloader),
          optimizer_params={'lr': 2e-5}
          )
```

## 4. Elasticsearch와의 연동
### 4-1. KNN search
유저 쿼리를 embedding model에 input으로 넣어서 예시와 같은 embedding vector를 얻고, 이미 embedding되어 Elasticsearch에 업로드 되어있는 도서 데이터들과 cosine similarity를 계산하여 가장 유사도가 높은 벡터를 찾아온다. 유저 쿼리와 유사도가 높다는 것은 유저의 니즈에 부합하는 책일 것이기 때문에, 이렇게 나온 결과로 책을 추천할 수 있다. 위 방식은 KR-SBERT를 사용하든, Contrastive Learning 등의 추가 학습을 통해 finetuning한 Custom embedding model이든 상관 없이 동일하게 적용된다.   
그러므로 만약 사용하는 embedding model을 바꾸고 싶다면 두 가지를 진행해야 한다. 먼저 Elasticsearch에 업로드 되어있는 도서 데이터 embedding vector를 사용하고자 하는 모델로 다시 embedding 하여 바꾸고 재 업로드 한 뒤에 KNN search의 과정에서도 해당 embedding model로 유저 쿼리를 임베딩 하여 KNN을 진행하면 된다.   

먼저 Elasticsearch에 업로드된 도서 데이터의 embedding을 바꾸기 위해서 sentence_transformer_encoding.py 을 변경하여 실행하고, 새로운 index에 데이터를 업로드 해야 한다.
index의 이름을 이전에 존재하지 않던 index의 이름으로 설정한 뒤, 새롭게 사용하고자 하는 embedding model을 불러와 모든 도서 데이터를 embedding 하고, 업로드한다. 이 과정이 끝나면 새로운 embedding model로 knn을 하기 위한 기본 세팅이 끝난다.   
(Book-rec-with-LLM-refactored/elasticsearch_upload/sentence_transformer_encoding.py의 4, 25, 83, 98 line 수정)

이렇게 custom model을 사용하기 위해서는 model 폴더를 실행시 폴더의 위치에 맞추어 model_name을 적절히 사용해야 한다. 혹은 실행 폴더와 상관 없이 제대로 된 모델을 사용하기 위해서는 model_name을 모델의 절대 경로로 설정하면 된다. 
```python
model = SentenceTransformer("knp_SIMCSE_unsup_128_v1-2023-11-15_16-54-15")

index_name = "new_data"

if es.indices.exists(index="new_data"):
    print("exists!")
    pass
else:
    es.indices.create(index=index_name, body={"mappings": mapping, "settings": setting})

def appendbulk(row):
    targetstring = f"category: {row['category']}, author: {row['author']}, introduction: {row['introduction']}, title: {row['title']}"

    embedding = model.encode(targetstring)
    data.append(
        {
            "_index": "new_data",
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
        }
    )
```

아래 코드가 Elasticsearch에서 유저의 질의가 들어왔을 때 embedding vector로 변환하는 부분이다. 이때 원래 사용하던 KR-SBERT의 이름을 넣는 것이 아닌 원하는 모델 이름을 넣어주면 해당 모델로 embedding을 진행할 수 있다.   
(Book-rec-with-LLM-refactored/elasticsearch_util/elasticsearch_retriever.py 89 line의 함수 수정)

이렇게 다른 모델을 사용하기 위해서는 위에서 언급한 것처럼 해당 모델 폴더를 model_name에서 명시된 폴더에 위치시켜야 하고, 모델 폴더의 위치에 상관 없이 모델을 사용하기 위해서는 model_name을 모델 폴더의 절대 경로로 설정하면 된다.
```python
def search_with_query(self, query: str) -> List[Bookdata]:
    with open("config.json", encoding="UTF-8") as f:
        config = json.load(f)
    n = config["elasticsearch_result_count"]

    model = SentenceTransformer("knp_SIMCSE_unsup_128_v1-2023-11-15_16-54-15")
    embed = model.encode(query)
```

위와 같이 KNN의 embedding model을 바꾸기 위해 실행해야 하는 코드는   
Book-rec-with-LLM-refactored/SBERT/sentence_transformer_encoding_custom_model.py,   
Book-rec-with-LLM-refactored/SBERT/elasticsearch_retriever_custom_model.py   
파일을 통해 업로드 되어있으니 참고할 수 있다.

### 작성자
| name | github_id |
| --- | :---: |
| 정성원 | `hungrymozzi` |
