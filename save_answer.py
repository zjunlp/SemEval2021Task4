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

from get_bad_cases import eval_model

import logging

logger = logging.getLogger(__name__)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_dataloader(tokenizer, args):
    if args.sliding_window:
        eval_dataset = (
            MultipleChoiceSlidingDataset(
                data_dir=args.data_dir,
                tokenizer=tokenizer,
                task=args.task_name,
                max_seq_length=args.max_seq_length,
                overwrite_cache=args.overwrite_cache,
                mode=Split.test,
            )
        )
    else:
        eval_dataset = MultipleChoiceDataset(
            data_dir=args.data_dir,
            tokenizer=tokenizer,
            task=args.task_name,
            max_seq_length=args.max_seq_length,
            overwrite_cache=args.overwrite_cache,
            mode=Split.test,
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

def save_answer(args, acc):
    answer = []
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
    res = {}
    res["answer"] = answer
    res["acc"] = acc
    if not os.path.exists(args.output_dir): os.mkdir(args.output_dir)
    with open(os.path.join(args.output_dir, 'result.pkl'), 'wb') as f:               #write
        pickle.dump(res, f)
    

def main():
    parser = argparse.ArgumentParser(description='hope it will work')
    parser.add_argument('--model_name_or_path', type=str, help='an integer for the accumulator')
    parser.add_argument('--data_dir', type=str, help='an integer for the accumulator', default='./dataset/task1')
    parser.add_argument('--max_seq_length', type=int, help='an integer for the accumulator', default=128)
    # parser.add_argument('--acc', type=float, help='an integer for the accumulator', default=80.4)
    parser.add_argument('--sliding_window', help='an integer for the accumulator', default=False,action="store_true" )
    parser.add_argument('--task_name',type=str,  help='an integer for the accumulator', default="semeval" )
    parser.add_argument('--output_dir',type=str,  help='an integer for the accumulator', default="./answer_file" )
    parser.add_argument('--overwrite_cache', help='overwrite_cache', default=False,action="store_true" )
    # parser.add_argument("--answer_list", nargs="+", default=["a", "b"], help="answer pickle")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    args = parser.parse_args()
    logger.warning("验证模型在验证集合上是否真实有效")
    acc = eval_model(args)
    # import IPython; IPython.embed(); exit(1)
    save_answer(args, acc)
    print("finish model dev acc: {},\n saved file path: {}".format(acc, args.output_dir))



if __name__ == "__main__":
    main()