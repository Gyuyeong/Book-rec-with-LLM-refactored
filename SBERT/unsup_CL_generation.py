from sentence_transformers import SentenceTransformer
import numpy as np

def embedding_sentence(sentence, model_name):
    # CL 학습 완료한 SBERT 모델 로드
    model = SentenceTransformer(model_name)

    # 모델을 이용해 embedding 진행
    embedding = model.encode(sentence, convert_to_tensor=True).cpu().numpy()

    return embedding

# embedding에 사용할 모델명 설정
model_name = 'C:/학교/23 산학협력프로젝트/모델 저장 파일/knp_SIMCSE_unsup_512_v3-2023-12-04_19-03-48'

# embedding할 문장 입력받기
sentence = input("임베딩할 문장 입력 : ")

# 문장 embedding 진행
embedding = embedding_sentence(sentence, model_name)

# 원래 문장과 embedding된 문장 출력
print("original sentence :", sentence)
print("embedding sentence :", embedding)