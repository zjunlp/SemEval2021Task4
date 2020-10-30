from torch.utils.data import dataloader
from transformers.configuration_auto import AutoConfig
from utils.multiple_choices import MultipleChoiceDataset
from transformers import AutoTokenizer, AutoModelForMultipleChoice, WEIGHTS_NAME, RobertaForMultipleChoice
import torch
from torch.utils.data import DataLoader, SequentialSampler
from utils import processors, MultipleChoiceDataset, Split
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
import argparse

from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


parser = argparse.ArgumentParser(description='Process some integers.')
path = './output/roberta-base_enhanced/checkpoint-14000'
parser.add_argument('--model_name_or_path', type=str, help='an integer for the accumulator', default=path)
parser.add_argument('--data_dir', type=str, help='an integer for the accumulator', default='./dataset/training_data')

args = parser.parse_args()

device = 'cuda'
config = AutoConfig.from_pretrained(
    args.model_name_or_path,
    num_labels = 6,
)
model = AutoModelForMultipleChoice.from_pretrained(args.model_name_or_path, config = config).to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained('/home/xx/pretrained_model/roberta-base')

eval_dataset = MultipleChoiceDataset(
    data_dir=args.data_dir,
    tokenizer=tokenizer,
    task='semeval',
    max_seq_length=512,
    overwrite_cache=False,
    mode=Split.dev,
)

eval_dataloader = DataLoader(
    eval_dataset,
    sampler=SequentialSampler(eval_dataset),
    batch_size=8,
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

def eval_step(inputs, ):
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





cnt = 0
bad_cases = []
t = []

# 开始输出
for inputs in tqdm(eval_dataloader):
    logits, labels = eval_step(inputs)
    logits = torch.argmax(logits,dim = 1)
    t += (logits == labels).tolist()

    bad_cases += logits.tolist()


total = len(t)
cor = 0.0
for a in t:
    if a:
        cor+=1
print('eval_acc:' + str(cor/total))


import pickle
import os
with open(os.path.join(args.model_name_or_path, 'eval_rrr'), 'wb') as writer:
    pickle.dump(bad_cases, writer)




