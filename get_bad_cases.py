from torch.utils.data import dataloader
import json
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
default_path = './output/roberta-base_enhanced/checkpoint-14000'
parser.add_argument('--model_name_or_path', type=str, help='an integer for the accumulator', default=default_path)
parser.add_argument('--data_dir', type=str, help='an integer for the accumulator', default='./dataset/training_data')
parser.add_argument('--max_seq_length', type=int, help='an integer for the accumulator', default='./dataset/training_data')

args = parser.parse_args()

device = 'cuda'
config = AutoConfig.from_pretrained(
    args.model_name_or_path,
    num_labels = 6,
)
model = AutoModelForMultipleChoice.from_pretrained(args.model_name_or_path, config = config).to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

eval_dataset = MultipleChoiceDataset(
    data_dir=args.data_dir,
    tokenizer=tokenizer,
    task='semeval',
    max_seq_length=args.max_seq_length,
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
softm = torch.nn.Softmax(dim = 1)

# 开始输出
for inputs in tqdm(eval_dataloader):
    logits, labels = eval_step(inputs)
    logits_temp = torch.argmax(logits,dim = 1)
    t += (logits_temp == labels).tolist()
    bad_cases += softm(logits).tolist()

total = len(t)
cor = 0.0
for a in t:
    if a:
        cor+=1
print('eval_acc:' + str(cor/total))

# import pickle
import os
# with open(os.path.join(args.model_name_or_path, 'eval_rrr'), 'wb') as writer:
#     pickle.dump(bad_cases, writer)

wrong_list = []

with open(os.path.join(args.data_dir,'dev.jsonl') ,'r', encoding='UTF-8') as reader:
    for line_id,line in enumerate(reader.readlines()):
        t = json.loads(line)
        op = torch.argmax(torch.tensor(bad_cases[line_id]), dim=0).item()
        if op != t['label']:
            t['wrong_label'] = op
            t['logits'] = bad_cases[line_id]
            wrong_list.append(t)


with open(os.path.join(args.model_name_or_path,'wrong_answer.json'), 'a', encoding='UTF-8') as writer:
    writer.writelines('\n-------------' +args.data_dir + '----------\n')
    for w in wrong_list:
        writer.writelines(json.dumps(w) + '\n')



