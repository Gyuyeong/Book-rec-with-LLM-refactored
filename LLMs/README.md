# 생성형 모델

의도 분류, 추천 도서 평가, 도서 추천 사유 생성에 사용되는 생성형 모델에 관한 문서.

## Requirements
Python 3.8.13

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

### 주의 사항
`Pytorch`와 `transformers`의 버전은 반드시 위 버전을 따라야 함
