import os
import csv
import enum
from torch.utils.data import dataloader
import json
from transformers.configuration_auto import AutoConfig
from utils.multiple_choices import MultipleChoiceDataset
from transformers import AutoTokenizer, AutoModelForMultipleChoice, WEIGHTS_NAME, RobertaForMultipleChoice
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from utils import processors, MultipleChoiceDataset, Split, MultipleChoiceSlidingDataset
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
import argparse
import numpy as np

from tqdm import tqdm, trange
from typing import Any, Callable, Dict, List, Optional, Tuple, Union



device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = AutoModelForMultipleChoice.from_pretrained(args.model_name_or_path).to(device)
# model.eval()
# albert_path = "/home/xx/pretrained_model/albert-xxlarge-v2"
# tokenizer = AutoTokenizer.from_pretrained(albert_path)


# only need path
tokenizer_path = {"albert" : "/home/xx/pretrained_model/albert-xxlarge-v2", "roberta": "/home/xx/pretrained_model/roberta-large", "xlnet":"/home/xx/pretrained_model/xlnet-large-cased"}

def judge_model(model_name):
    if "albert" in model_name:
        return "albert"
    elif "roberta" in model_name:
        return "roberta"
    elif "xlnet" in model_name:
        return "xlnet"
    else:
        assert 1 == 2, "no model in model_name"

def get_dataloader(tokenizer, args):
    """
    由于GPU限制，所以batch_size=1，免得占太多显存，eval慢点无所谓。
    """
    if args.sliding_window:
        eval_dataset = (
            MultipleChoiceSlidingDataset(
                data_dir=args.data_dir,
                tokenizer=tokenizer,
                task=args.task_name,
                max_seq_length=args.max_seq_length,
                overwrite_cache=args.overwrite_cache,
                mode=Split.dev,
            )
        )
    else:
        eval_dataset = MultipleChoiceDataset(
            data_dir=args.data_dir,
            tokenizer=tokenizer,
            task=args.task_name,
            max_seq_length=args.max_seq_length,
            overwrite_cache=args.overwrite_cache,
            mode=Split.dev,
        )

    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=SequentialSampler(eval_dataset),
        batch_size=1,
        drop_last=False,
        collate_fn=default_data_collator,
        num_workers=1,
    )
    return eval_dataloader

def _prepare_inputs(inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
    """
    Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    return inputs

# def eval_step(inputs,):
#     inputs = _prepare_inputs(inputs)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs[1:]
    
#     logits = tuple(logit.detach() for logit in logits)
#     if len(logits) == 1:
#         logits = logits[0]
    
#     labels = tuple(inputs.get(name).detach() for name in ['labels'])
#     if len(labels) == 1:
#         labels = labels[0]

#     return (logits, labels)

def count_answer(answer):
    """
    answer : List[int]
    return : 出现最多的数字 means the answer
    """
    assert answer.shape[0] > 0
    res = np.argmax(np.bincount(answer))
    return res



import pickle
import numpy as np
def model_ensemble_offline(file_list):
    """
    通过输入的pickle dumps 文件，将每一个模型的输出混合起来
    pickle 是一个二维list 存放着每一个sample的logits分布。 t_answer: List[List[float]]

    """
    answer = np.array([])
    for file in file_list:
        with open(file, 'rb') as reader:
            t_answer = pickle.load(reader)
        if answer.shape[0] == 0:
            answer = np.array(t_answer, dtype=np.float)
        else:
            answer += np.array(t_answer, dtype=np.float)
    answer = np.argmax(answer, axis=1)


def eval_model(args, model, tokenizer):
    eval_dataloader = get_dataloader(tokenizer, args)
    preds = get_answer(model, eval_dataloader)
    preds = torch.argmax(torch.tensor(preds),dim=1).cpu().numpy()
    labels = get_labels(args)

    return compute_acc(preds, labels)

    


def model_ensemble_online(args):
    answer = np.array([])
    model_path = args.model_name_or_path

    model = AutoModelForMultipleChoice.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path[judge_model(model_path)])
    eval_dataloader = get_dataloader(tokenizer, args)
    if answer.shape[0] == 0:
        answer = get_answer(model, eval_dataloader)
    else:
        answer += get_answer(model, eval_dataloader)
    answer = np.argmax(answer, axis=1)

    return answer



def compute_acc(preds, labels):
    return (preds == labels).mean()

def get_labels(args):
    """
    依次读取label
    """
    labels = []
    dev_path = args.data_dir
    with open(os.path.join(dev_path,"dev.jsonl"), "r", encoding='UTF-8') as reader:
        for line in reader.readlines():
            t = json.loads(line)
            labels.append(t['label'])

    return np.array(labels, dtype=np.int)


def get_answer(model, dataloader):
    model.eval()
    answer = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=''):
            batch = _prepare_inputs(batch)
            _, output = model(**batch)
            output = F.softmax(output, dim=1)
            answer += output
        answer = torch.stack(answer, dim=0)

    return answer.cpu().numpy()
# import pickle


import pandas as pd
def write_answer_to_file(answer, args):
    name = "subtask1.csv" if "task1" in args.data_dir else "subtask2.csv"
    file_path = os.path.join("./answer_file", name)
    # turn to Int
    answer = answer.astype(int)
    b = pd.DataFrame(answer, columns=['a']).astype(int)
    b.to_csv(file_path, header=0)
    # import IPython; IPython.embed(); exit(1)
    # np.savetxt(file_path, answer, delimiter=",")






def main():

    parser = argparse.ArgumentParser(description='hope it will work')
    default_path = './output/roberta-base_enhanced/checkpoint-14000'
    parser.add_argument('--model_name_or_path', type=str, help='an integer for the accumulator', default=default_path)
    parser.add_argument('--data_dir', type=str, help='an integer for the accumulator', default='./dataset/training_data')
    parser.add_argument('--max_seq_length', type=int, help='an integer for the accumulator', default='./dataset/training_data')
    parser.add_argument('--sliding_window', help='an integer for the accumulator', default=False,action="store_true" )
    parser.add_argument('--task_name',type=str,  help='an integer for the accumulator', default="semeval" )
    parser.add_argument('--overwrite_cache', help='overwrite_cache', default=False,action="store_true" )
    parser.add_argument("--answer_list", nargs="+", default=["a", "b"], help="answer pickle")
    parser.add_argument("--model_list", nargs="+", default=["a", "b"], help="model list")

    args = parser.parse_args()
    preds =model_ensemble_online(args.model_list)
    # labels = get_labels(os.path.join(args.data_dir, "dev.jsonl"))
    write_answer_to_file(preds, args)

if __name__ == "__main__":
    main()


