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

# configuration
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "EVALUATION: 주어진 사용자의 질문과 정보를 바탕으로 한 책 추천의 적절성을 평가해 주세요. "
        + "만약 추천된 책의 INFO가 사용자의 QUERY와 관련되어 있고 사용자의 요청에 잘 부합한다면 'Pass’를, "
        + "그렇지 않다면 'Fail'을 부여하세요. "
        + "책의 주제, 사용자 질문과의 관련성, 그리고 사용자의 필요와 의도에 대한 적합성을 고려하세요. "
        + "추천이 적절하고 관련이 있는지 평가하는 것이 중요합니다.\n\n"
    )
}

with open("./pgwdata300.json", "r", encoding="utf-8-sig") as json_file:
    list_data_dict = json.load(json_file)
print(list_data_dict[0])


class SFT_dataset(Dataset):
    def __init__(
        self,
        data_path_1_SFT: str,
        tokenizer: transformers.PreTrainedTokenizer,
        verbose=False,
    ):
        super(SFT_dataset, self).__init__()

        # Open JSON data
        with open(data_path_1_SFT, "r", encoding="utf-8-sig") as json_file:
            list_data_dict = json.load(json_file)
            if verbose:
                print("## First Data Sample ##")
                print(list_data_dict[0])

        # Processing the data
        sources = []
        targets = []
        for example in list_data_dict:
            # Extract and format the prompt
            prompt_template = example["prompt"]
            input_text = example["input"]
            formatted_prompt = prompt_template.replace(
                PROMPT_DICT["prompt_input"], input_text
            )
            sources.append(formatted_prompt)

            # Append completion with eos_token
            eos_token = tokenizer.eos_token
            completion_text = example["completion"] + eos_token
            targets.append(completion_text)

            if verbose and len(sources) == 1:
                print("Example Source:", sources[0])
                print("Example Target:", targets[0])
                print("Tokenizing inputs... This may take some time...")

        # Combine source and target for examples
        examples = [s + t for s, t in zip(sources, targets)]

        # Tokenization
        sources_tokenized = self._tokenize_fn(sources, tokenizer)
        examples_tokenized = self._tokenize_fn(examples, tokenizer)

        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX

        self.input_ids = input_ids
        self.labels = labels
        logging.warning("Loading data done!!: %d" % len(self.labels))

    def _tokenize_fn(
        self, strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
    ) -> Dict:
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings
        ]

        input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]

        return dict(input_ids=input_ids, input_ids_lens=input_ids_lens)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


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


# Prepare data
train_dataset = SFT_dataset(data_path_1_SFT="./pgwdata300.json", tokenizer=tokenizer)
eval_dataset = None  # no evaluation
data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

# prepare LoRA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05, bias="none"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

# parameters for training the model
training_args = TrainingArguments(
    output_dir="./pgwfolder2/pgwmodel300_lr_1e-10",  # The output directory
    overwrite_output_dir=True,  # overwrite the content of the output directory
    num_train_epochs=300,  # number of training epochs
    per_device_train_batch_size=1,  # batch size for training
    per_device_eval_batch_size=1,  # batch size for evaluation
    eval_steps=3,  # Number of update steps between two evaluations.
    save_steps=10,  # after # steps model is saved
    logging_steps=10,
    warmup_steps=5,  # number of warmup steps for learning rate scheduler
    prediction_loss_only=True,
    fp16=True,
    gradient_accumulation_steps=16,
    learning_rate=1e-10,
    lr_scheduler_type="cosine",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()  # train
