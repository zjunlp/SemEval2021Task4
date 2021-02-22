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
        batch_size=4,
        drop_last=False,
        collate_fn=default_data_collator,
        num_workers=4,
    )
    # from IPython import embed;embed();exit(1)
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


import pickle
import numpy as np

    
def compute_acc(len_answer, labels,left,right):
    total_num = right - left + 1
    acc = 0
    while left <= right:
        acc += (len_answer[left][1] == labels[left][1]) / total_num
        left += 1
    return acc

    

def get_labels(args):
    """
    依次读取label
    """
    labels = []
    dev_path = args.data_dir
    with open(os.path.join(dev_path,"dev.jsonl"), "r", encoding='UTF-8') as reader:
        for line in reader.readlines():
            t = json.loads(line)
            length = len(t["article"].split()) 
            labels.append([length,t['label']])
    labels.sort()
    return labels

# import pickle


def get_len_answer(args):
    answer,len_answer = [],[]
    model = AutoModelForMultipleChoice.from_pretrained(args.model_name_or_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path) 
    eval_dataloader = get_dataloader(tokenizer, args)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc=''):
            batch = _prepare_inputs(batch)
            output= model(**batch)[1]
            output = F.softmax(output, dim=1)
            answer += output
    answer = torch.stack(answer, dim=0)
    answer = answer.cpu().numpy()
    answer = torch.argmax(torch.tensor(answer),dim=1).cpu().numpy()
    id = 0
    with open(os.path.join(args.data_dir,'dev.jsonl'),'r') as lines:
        for line in lines.readlines():
            data = json.loads(line)
            length = len(data["article"].split()) 
            len_answer.append([length,answer[id]])
            id += 1
    # print(len_answer)
    len_answer.sort()
    # from IPython import embed;embed()
    # from IPython import embed;embed();exit(1)
    return len_answer

def print_result(args):
    result = []
    len_answer = get_len_answer(args)
    labels = get_labels(args)
    right_index = [340,680,1020,1360]
    # right_index = [364,688,1012,1336]
    passenge_nums = []
    right = -1
    for index in right_index:
        left = right + 1
        right = max([i for i,x in enumerate(len_answer) if x[0] <= index])
        acc = compute_acc(len_answer,labels,left,right)
        result.append(acc)
        passenge_nums.append(right -left + 1)
    left = right + 1
    acc = compute_acc(len_answer,labels,left,len(len_answer) - 1)
    passenge_nums.append(len(len_answer) -left)
    result.append(acc)
    print(result,passenge_nums)


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
    parser.add_argument("--model_path",type=str,default="/home/chenxn/SemEval2021/pretrained_model/xlnet-large-cased")

    args = parser.parse_args()
    print_result(args)

if __name__ == "__main__":
    main()


