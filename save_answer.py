import os
import csv
import enum
from torch.utils.data import dataloader
import json
from transformers.configuration_auto import AutoConfig
# from utils.multiple_choices import MultipleChoiceDataset
from transformers import AutoTokenizer, AutoModelForMultipleChoice, WEIGHTS_NAME, RobertaForMultipleChoice
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from utils import processors, MultipleChoiceDataset, Split, MultipleChoiceSlidingDataset
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
import argparse
import numpy as np
from IPython import embed 
from tqdm import tqdm, trange
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pickle
import numpy as np


parser = argparse.ArgumentParser(description='hope it will work')
# default_path = './output/roberta-base_enhanced/checkpoint-14000'
default_path = './output/xlnet-large-cased_task1_accumulate16_polylr8e-6/checkpoint-2250'
parser.add_argument('--model_name_or_path', type=str, help='an integer for the accumulator', default=default_path)
parser.add_argument('--data_dir', type=str, help='an integer for the accumulator', default='./dataset/training_data')
parser.add_argument('--max_seq_length', type=int, help='an integer for the accumulator', default='./dataset/training_data')
parser.add_argument('--sliding_window', help='an integer for the accumulator', default=False,action="store_true" )
parser.add_argument('--task_name',type=str,  help='an integer for the accumulator', default="semeval" )
parser.add_argument('--overwrite_cache', help='overwrite_cache', default=False,action="store_true" )
parser.add_argument("--answer_list", nargs="+", default=["a", "b"], help="answer pickle")
parser.add_argument("--model_list", nargs="+", default=["a", "b"], help="model list")

args = parser.parse_args()

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
tokenizer_path = {"albert" : "/home/xx/pretrained_model/albert-xxlarge-v2", "roberta": "/home/xx/pretrained_model/roberta-large", "xlnet":"/home/chenxn/SemEval2021/pretrained_model/xlnet-large-cased"}

def judge_model(model_name):
    if "albert" in model_name:
        return "albert"
    elif "roberta" in model_name:
        return "roberta"
    elif "xlnet" in model_name:
        return "xlnet"
    else:
        assert 1 == 2, "no model in model_name"

def get_dataloader(tokenizer):
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
        num_workers=8,
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

def save_answer(model_list):
    answer=[]
    for model_path in model_list:
        model = AutoModelForMultipleChoice.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path[judge_model(model_path)])
        eval_dataloader = get_dataloader(tokenizer)
        model.eval()
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc=''):
                batch = _prepare_inputs(batch)
                output= model(**batch)[1]
                output = F.softmax(output, dim=1)
                answer += output
                break
    answer = torch.stack(answer, dim=0)
    answer=np.array(answer)
    with open('result.pkl', 'wb') as f:               #write
        pickle.dump(answer, f)
        f.close()
    

def main():
    save_answer(args.model_list)
    print("finish")



if __name__ == "__main__":
    main()