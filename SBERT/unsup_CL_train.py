from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import os
import gzip
import csv
import pandas as pd

# 디버그 메시지 로그를 stdout으로 작성해주는 부분
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# huggingface/transformers의 pre-trained model을 사용할 수 있음. ex) bert-base-uncased, roberta-base, xlm-roberta-base
# 우리는 한국어 특화 SBERT 모델인 KR-SBERT를 사용
model_name = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'

# 학습 모델의특징이나 버전등을 알 수 있도록 적절한 모델 네임, 경로 설정
model_save_path = 'knp_SIMCSE_unsup_512_v4_lr1e-4-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# 학습시 최종 모델이 아닌 중간에도 checkpoint마다 모델을 저장하여 성능을 체크할 수있는 기능
# 얼마마다 checkpoint를 저장할지도 뒤에서 설정 가능
checkpoint_path = model_save_path + '_checkpoint'

# Token을 embedding으로 매핑해줄 수 있는 huggingface/transformers의 모델을 설정
# 위에서 가져온 model name을 기준으로 선택됨
word_embedding_model = models.Transformer(model_name, max_seq_length=512)

# mean pooling을 통해 하나의 fixed size sentence vector를 얻음
# pooling 모델은 꼭 mean pooling이 아니라 다른 것으로 바꿔서도 사용할 수 있음
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

# 위에서 설정한 embedding model과 pooling model을 넣어 실제 학습할 sentencetransformer 모델 설정
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# 학습 데이터는 공앤박의 도서 데이터에서 추천시 필요할 것으로 생각되는 카테고리, 저자, 소개, 제목의 네 가지 정보만을 빼서 만듦
# 만들어진 데이터를 이용해 모델을 unsupervised learning 방식으로 학습시킬 것이기 때문에
# 하나의 batch 안에 들어갈 데이터가 서로 negative pair가 될 수 있도록 도서의 카테고리를 기준으로 조정
data = pd.read_csv("unsupCLdata_shuffled_v3_212k.csv")

# 'book_info' 열에서 한 행씩 읽어서 train_samples(학습데이터) 리스트에 추가
# unsupervised Contrastive Learning을 위해 동일 데이터를 두 개씩 묶은 데이터가 들어가게 됨
train_samples = []
for index, row in data.iterrows():
    book_info = row['book_info']
    train_samples.append(InputExample(texts=[book_info, book_info]))

# batch size 및 epoch 설정
# 성능 비교를 위해 다양한 값으로 바꾸어 학습 가능
train_batch_size = 16
num_epochs = 10

# Contrastive Learning을 위해 MultipleNegativesRankingLoss 사용
train_dataloader = DataLoader(train_samples, shuffle=False, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model)

# 전체 학습 스텝의 10%를 warmup step으로 사용
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
logging.info("Warmup-steps: {}".format(warmup_steps))

# 학습 진행
# 아래의 파라미터는 다양하게 바꿀 수 있음
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          checkpoint_path=checkpoint_path,
          checkpoint_save_steps=len(train_dataloader),
          optimizer_params={'lr': 2e-5}
          )