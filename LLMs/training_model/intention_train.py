import pandas as pd
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from transformers import pipeline
from peft import LoraConfig, get_peft_model, TaskType
import json
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import copy
from copy import deepcopy


"""
JSON 형태의 학습 데이터를 학습이 가능한 torch.utils.data.Dataset 형으로 변환하기 위해 정의한 class
Intention 학습을 위한 각 JSON 데이터에는 input, clues, reasoning, 그리고 intention이 있음.
이 중 input은 모델에게 주어지는 것이고, clues, reasoning, intention은 모델이 생성해야 하는 label들이다.
각 데이터에 전처리를 거쳐서 모델의 학습 성능을 높일 수 있는 형태로 만들어주는 과정.

torch.utils.data.Dataset을 inherit하기 위해서 필수적으로 constructor와 __len__, __getitem__을 구현해야 함
"""
class SFT_dataset(Dataset):
    """
    Constructor
    @params:
    data_path_1_SFT: str: JSON 데이터 경로
    tokenizer: transformers.PretrainedTokenizer: 모델의 tokenizer. 이 tokenizer로 학습 데이터를 토큰화
    verbose: boolean: True일 경우, 데이터 하나를 보여줌
    """
    def __init__(self, data_path_1_SFT: str, tokenizer: transformers.PreTrainedTokenizer, verbose=False):
        super(SFT_dataset, self).__init__()
        
        # open json data
        with open(data_path_1_SFT, "r", encoding='utf-8-sig') as json_file:
            list_data_dict = json.load(json_file)
            if verbose:  # set verbose = True to check the first data loaded from json
                print('## data check ##')
                print((list_data_dict[0]))
                
        prompt_input = PROMPT_DICT["prompt_input"]  # get template
        
        # inputs: 모델이 생성을 위해 받는 입력. 사용자가 chatbot에 질의한 문장이 들어오게 된다
        sources = []
        for example in list_data_dict:
            tmp = prompt_input.format_map({"input": example["input"]})  # 입력 문장 앞에 prompt를 붙인다
            sources.append(tmp)
        
        # outputs: 모델이 생성해야 하는 것
        # 각각 단서, 추론, 의도를 한꺼번에 생성하도록 데이터를 준비
        # 중간에 CLUE_TOKEN, REASONING_TOKEN, LABEL_TOKEN이 포함되어 있음
        # TOKEN을 추가해서 모델이 생성할 때 해당 토큰들을 포함해서 생성하게끔 유도하는 역할
        # 이렇게 하면 모델이 생성시 clue와 reasoning, 그리고 label이 누락되지 않을 확률이 높아질 것으로 기대
        targets = []
        for example in list_data_dict:
            targets.append(f"{CLUE_TOKEN}단서들: {example['clues']}{CLUE_TOKEN}\n{REASONING_TOKEN}추론: {example['reasoning']}{REASONING_TOKEN}\n{LABEL_TOKEN}의도: {example['intention']}{LABEL_TOKEN}{tokenizer.eos_token}")
            
        if verbose:
            idx = 0
            print((sources[idx]))
            print((targets[idx]))
            print("Tokenizing inputs... This may take some time...")

        examples = [s + t for s, t in zip(sources, targets)]  # concatenate source and target
        
        # source와 example들을 tokenize
        sources_tokenized = self._tokenize_fn(sources, tokenizer)
        examples_tokenized = self._tokenize_fn(examples, tokenizer)
        
        # examples_tokenized (source + target)를 복사해서 source 부분은 IGNORE_INDEX로 가려준다
        # label은 target 부분만 학습하기를 원한다
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):  # we train only on targets for labels,so cover the source part
            label[:source_len] = IGNORE_INDEX
        
        data_dict = dict(input_ids=input_ids, labels=labels)
        
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        logging.warning("Loading data done!!: %d"%(len(self.labels)))
        
    """
    주어진 sequence를 tokenize해주는 메소드
    @params:
    strings: Sequence[str]: string들이 들어있는 list와 같은 sequence
    tokenizer: transformers.PretrainedTokenizer: 사용할 tokenizer. 해당 class를 intialize 할 때 준 tokenizer를 사용함
    @return:
    tokenize 된 sequence들과 각 길이 정보가 들어간 dictionary 반환
    """
    def _tokenize_fn(self, strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
        # 각 string을 tokenize
        # pytorch tensor형태로 반환받음
        # 가장 긴 sequence를 기준으로 padding을 함
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True
            )
            for text in strings
        ]
        
        # Tokenize 된 결과와 그 길이를 저장
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens
        )
    
    # Dataset class inherit을 위해 반드시 구현해야 함
    def __len__(self):
        return len(self.input_ids)
    
    # Dataset class inherit을 위해 반드시 구현해야 함
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


"""
정적으로 padding을 실시하면 데이터를 전처리하는 시간이 너무 오래걸리기에
학습을 하는 와중에 그때그때 동적으로 padding을 해줄 수 있는 class
"""
@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
        
    def __call__(self, instances: Sequence[Dict]) -> Dict [str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # pad input sequence
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        # pad label sequence
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        )


#configuration
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
# 원활한 생성을 위해 추가한 custom token들
CLUE_TOKEN = "<CLUE>"
REASONING_TOKEN = "<REASONING>"
LABEL_TOKEN = "<LABEL>"
# 모델에게 해야할 일을 알려주는 prompt
# 각 학습 데이터마다 해당 prompt가 앞에 붙는다
PROMPT_DICT = {
    "prompt_input": (
        "이 모델은 '입력 문장'을 분석해서 의도를 분류하는 모델입니다. " + 
        "분류할 의도로는 '메타 정보 검색', '키워드 검색', '유사도서 검색', '판매량 검색', 그리고 '그 외'가 있습니다. " + 
        "입력 문장이 책 추천과 관련이 없는 경우 '그 외'로 분류합니다. " + 
        "입력 문장이 책 추천과 관련이 있는 경우, 다음과 같이 분류합니다. " + 
        "저자, 출판사, 기간 등 메타 정보가 포함된 경우 '메타 정보 검색'으로 분류합니다. " + 
        "키워드 정보만 포함된 경우 '키워드 검색'으로 분류합니다. " + 
        "제목 정보가 포함된 경우 예외 없이 '유사도서 검색'으로 분류합니다. " + 
        "판매량 정보가 들어간 경우 예외 없이 '판매량 검색'으로 분류합니다. " + 
        "우선, 입력 데이터를 받으면 주어진 입력 데이터에서 의도를 분류할 때 도움이 될 수 있는 단서들을 추출합니다. " + 
        "그리고 각 단서마다 어떤 종류의 단서인지를 추가합니다. " + 
        "그 다음, 주어진 입력 데이터와 단서들을 바탕으로 어떤 의도인지 추론하는 글을 생성합니다. " + 
        "그 후, 입력 데이터와 단서들과 추론한 글을 바탕으로 의도를 분류합니다. " +
	    "반드시 '단서들', '추론', '의도' 순서대로 생성해야 합니디. " +
	    "의도를 추론할 때 반드시 주어진 의도를 문자 그대로 생성해야 합니다.\n"
	    "입력: {input}\n\n"
    )
}

# get model and tokenizer
model = AutoModelForCausalLM.from_pretrained("rycont/kakaobrain__kogpt-6b-8bit")
tokenizer = AutoTokenizer.from_pretrained("rycont/kakaobrain__kogpt-6b-8bit",
                                         padding_side="right",
                                         model_max_length=1024)  # input의 최대 길이는 1024
tokenizer.add_special_tokens(
    {
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
    }
)
tokenizer.pad_token = tokenizer.eos_token
# custom token들 추가하는 과정 
tokenizer.add_tokens([CLUE_TOKEN, REASONING_TOKEN, LABEL_TOKEN], special_tokens=True)
model.resize_token_embeddings(len(tokenizer)) # token 이 추가되었으니, model의 embedding 크기를 다시 맞춰춰야 한다.


# Prepare data
train_dataset = SFT_dataset(data_path_1_SFT="./ner_tagged_data.json", tokenizer=tokenizer)
eval_dataset = None  # no evaluation
data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

# prepare LoRA
# 각 hyperparameter들은 여러 논문 밑 경험상 가장 괜찮았던 값들로 구성함
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, config) # 기본 모델에 LoRA Adapter가 추가되는 형식으로 학습이 진행된다

model.print_trainable_parameters()


training_args = TrainingArguments(
    output_dir="./intention_models/lora_results/kakao_models/just_intention_title_brackets_n_100_lr_3e_5", # 학습된 모델 저장 경로
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=100, # number of training epochs
    per_device_train_batch_size=1, # batch size for training. GPU 크기 문제로 batch size는 1로 해야 했다
    per_device_eval_batch_size=1,  # batch size for evaluation
    eval_steps = 3, # Number of update steps between two evaluations.
    save_steps=10, # after # steps model is saved
    logging_steps=10,
    warmup_steps=5,# number of warmup steps for learning rate scheduler
    prediction_loss_only=True,
    fp16=True,  # 공간 절약을 위해 floating point 16d으로 학습을 진행 (물론 현재 사용하는 모델이 fp16이라 큰 상관은 없다)
    gradient_accumulation_steps=16,  # GPU 공간 절약을 위해 추가한 hyperparameter
    learning_rate=3e-5,  # learning rate은 조금 작게 하고 오랫동안 학습시키는 방식을 택함 (1e-05 3e-05 테스트 해봄)
    lr_scheduler_type="cosine"  # learning rate가 시간이 갈수록 점점 작아지게끔 하는 scheduler 사용
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()  # train
