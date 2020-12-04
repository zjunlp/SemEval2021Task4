from torch.utils.data import dataloader
import json
from transformers.configuration_auto import AutoConfig
from utils.multiple_choices import MultipleChoiceDataset
from transformers import AutoTokenizer, AutoModelForMultipleChoice, WEIGHTS_NAME, RobertaForMultipleChoice
import torch
from torch.utils.data import DataLoader, SequentialSampler
from utils import processors, MultipleChoiceDataset, Split, MultipleChoiceSlidingDataset
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
import argparse
import numpy as np

from tqdm import tqdm, trange
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


parser = argparse.ArgumentParser(description='hope it will work')
default_path = './output/roberta-base_enhanced/checkpoint-14000'
parser.add_argument('--model_name_or_path', type=str, help='an integer for the accumulator', default=default_path)
parser.add_argument('--data_dir', type=str, help='an integer for the accumulator', default='./dataset/training_data')
parser.add_argument('--max_seq_length', type=int, help='an integer for the accumulator', default='./dataset/training_data')
parser.add_argument('--sliding_window', help='an integer for the accumulator', default=False,action="store_true" )
parser.add_argument('--task_name',type=str,  help='an integer for the accumulator', default="semeval" )
parser.add_argument('--overwrite_cache', help='overwrite_cache', default=False,action="store_true" )

args = parser.parse_args()

device = 'cuda'
model = AutoModelForMultipleChoice.from_pretrained(args.model_name_or_path).to(device)
model.eval()
albert_path = "/home/xx/pretrained_model/albert-xxlarge-v2"
tokenizer = AutoTokenizer.from_pretrained(albert_path)

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
    num_workers=8,
)
def _prepare_inputs(inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
    """
    Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    return inputs

def eval_step(inputs,):
    inputs = _prepare_inputs(inputs)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs[1:]
    
    logits = tuple(logit.detach() for logit in logits)
    if len(logits) == 1:
        logits = logits[0]
    
    labels = tuple(inputs.get(name).detach() for name in ['labels'])
    if len(labels) == 1:
        labels = labels[0]

    return (logits, labels)

def count_answer(answer):
    """
    answer : List[int]
    return : 出现最多的数字 means the answer
    """
    assert answer.shape[0] > 0
    res = np.argmax(np.bincount(answer))
    return res

cnt = 0
bad_cases = []
t = []
softm = torch.nn.Softmax(dim = 1)
num_sample = len(eval_dataset)
"""
小心 num_sample > example_idx  因为每一个example分成了很多个sample
"""
answer_list = [None] * num_sample
# 开始输出
idx = 0
real_answer = [None] * num_sample
answer = []
#TODO 利用logits 而不是最多出现的答案来选取.
for inputs in tqdm(eval_dataloader):
    logits, labels = eval_step(inputs)
    logits_temp = torch.argmax(logits,dim = 1)
    example_idx = int(eval_dataset[idx].example_id)
    real_answer[example_idx] = labels.item()
    l = logits_temp.item()
    if answer_list[example_idx] is not None:
        answer_list[example_idx].append(l)
    else:
        answer_list[example_idx] = [l]
    t += (logits_temp == labels).tolist()
    bad_cases += softm(logits).tolist()
    idx += 1
for a in answer_list:
    if a is None:
        break
    answer.append(count_answer(np.array(a)))

# convert to ndarray to compute acc
if args.sliding_window:
    answer = np.array(answer)
    real_answer = np.array(real_answer)[:answer.shape[0]]

    acc = (answer == real_answer).mean()
    print("sliding window acc: " + str(acc))

import IPython; IPython.embed(); exit(1)
total = len(t)
cor = 0.0
for a in t:
    if a:
        cor+=1
print('eval_acc:' + str(cor/total))

# import pickle
def main():
    import os
    # with open(os.path.join(args.model_name_or_path, 'eval_rrr'), 'wb') as writer:
    #     pickle.dump(bad_cases, writer)

    wrong_list = []
    cnt = 0
    with open(os.path.join(args.data_dir,'dev.jsonl') ,'r', encoding='UTF-8') as reader:
        for line_id,line in enumerate(reader.readlines()):
            t = json.loads(line)
            op = torch.argmax(torch.tensor(bad_cases[line_id]), dim=0).item()
            if op != t['label']:
                t['wrong_label'] = op
                t['logits'] = bad_cases[line_id]
                wrong_list.append(t)
                cnt += 1

    assert cnt == total - cor

    with open(os.path.join(args.model_name_or_path,'wrong_answer.json'), 'w', encoding='UTF-8') as writer:
        for w in wrong_list:
            # ensure_ascii=False is important to avoid the luanma
            writer.writelines(json.dumps(w, ensure_ascii=False) + '\n')



