from logging import debug
import copy
from os import write
import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import csv
import numpy as np
import json

import pickle

from transformers import AutoTokenizer, AutoModelForMaskedLM, AlbertModel
from transformers import pipeline

model_name = "albert-xxlarge-v2"
model_name = "roberta-large"


tokenizer = AutoTokenizer.from_pretrained('/home/xx/pretrained_model/' + model_name)
model = AutoModelForMaskedLM.from_pretrained('/home/xx/pretrained_model/' + model_name, return_dict=True)



def save_csv(file_name, data):
    with open(file_name, 'w', encoding='UTF-8') as writer:
        tsv_w = csv.writer(writer)
        tsv_w.writerow(['article', 'question','label', 'option_0','option_1','option_2','option_3','option_4',
        'label'
        ])  # 单行写入
        tsv_w.writerows(data)  # 多行写入



data_path = './dataset/training_data/dev.jsonl'
# train_data = load_data(data_path)

def count_words(sen):
    cnt = 0
    for w in sen:
        if w == ' ':
            cnt += 1
    return cnt

unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)


# return : [batch_size, top5 words]
def top5_words(sentences ):
    temp_dict = unmasker(sentences)
    # only single sentence
    if len(temp_dict) != len(sentences):
        temp_dict = [temp_dict]

    result = [[] * len(sentences)]
    for idx, t in enumerate(temp_dict):
        for tt in t:
            result[idx].append(tokenizer.convert_tokens_to_string(tt['token_str']).strip())
    return result

def _prepare_inputs(inputs):
    """
    Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to('cuda')


    return inputs


# predict which option is mostly close to the MLM word
def get_token_output(sentence, options):
    t = []
    for op in options:
        ids = tokenizer(' ' + op, add_special_tokens=False, return_tensors='pt')['input_ids'][0].to('cuda')

        inputs = tokenizer(sentence.replace('@placeholder', op), return_tensors='pt', padding=True)

        inputs = _prepare_inputs(inputs)
        for i in ids:
            iid = torch.where(inputs['input_ids'] == i.item())[1][0].cpu().detach().item()

        outputs = model.roberta(**inputs)[0][0][iid]
        t.append(outputs.cpu().detach())
    logits = []
    for x in t[:-1]:
        logits.append(torch.cosine_similarity(x, t[-1], dim=0))
    res = torch.argmax(torch.tensor(logits), dim=0).item()

    return res

def add_new_option(data_path, new_file_name=None):
    if new_file_name == None:
        new_file_name = data_path.replace('train','train_enhanced') 
    new_dict = []
    total = 0
    correct = 0
    pretrain = []
    with open(data_path, 'r') as file:
        for line in tqdm(file.readlines()):
            t = json.loads(line)
            article = t['article']
            mask_sentence = article[:400] + t['question'].replace('@placeholder', '<mask>')
            options =[ t['option_0'], t['option_1'],t['option_2'],t['option_3'],t['option_4']]
            result = top5_words(mask_sentence)[0]
            pretrain.append(result)

            # pred = get_token_output(article[:400] + t['question'], options)
            # if pred == t['label']:
            #     correct += 1
            total += 1
            for w in result:
                if w not in options:
                    t['option_5'] = w
                    break
            res = copy.deepcopy(t)
            new_dict.append(res)
        with open(new_file_name, 'w') as writer:
            for d in new_dict:
                writer.writelines(json.dumps(d) + '\n')

            
    print(correct / total)
    import IPython; IPython.embed(); exit(1)

        
def add_new_option_1(data_path, new_file_name=None):
    if new_file_name == None:
        new_file_name = data_path.replace('.jsonl','') + 'enhanced' + '.jsonl'
    new_dict = []
    total = 0
    correct = 0
    with open(data_path, 'r') as file:
        for line in tqdm(file.readlines()):
            t = json.loads(line)
            article = t['article']
            mask_sentence = article[:400] + t['question'].replace('@placeholder', '<mask>')
            options =[ t['option_0'], t['option_1'],t['option_2'],t['option_3'],t['option_4']]
            result = top5_words(mask_sentence)[0]

            options.append(result[0])
            pred = get_token_output(article[:400] + t['question'], options)
            if pred == t['label']:
                correct += 1
            total += 1
            for w in result:
                if w not in options:
                    t['option_5'] = w
                    break
            res = copy.deepcopy(t)
            new_dict.append(res)
        with open(new_file_name, 'w') as writer:
            for d in new_dict:
                writer.writelines(json.dumps(d))

    print(correct / total)
    import IPython; IPython.embed(); exit(1)
        # len_max = min(count_words(t['article']), len_max)
        # if count_words(t['article']) > 400:
        #     cnt += 1

batch_size = 16

file_path = './dataset/enhanced_roberta_task2/train.jsonl'

# add_new_option(file_path)

# save answer file sorted by id
# label 0,1,2,3,4
def get_answer(file_path):
    answer_dict = {}
    answer_dict['labels'] = []
    with open(file_path, 'r') as file:
        for idx,line in enumerate(file.readlines()):
            t_json = json.loads(line)
            answer_dict['labels'].append(t_json['label'])

    with open(file_path.replace('.jsonl', '_answer.jsonl'), 'w') as writer:
        writer.write(json.dumps(answer_dict))

get_answer('./dataset/task1/dev.jsonl')
