import json
import pickle
import os


answers = []

with open('/home/xx/AI/SemEval2021-Reading-Comprehension-of-Abstract-Meaning/output/roberta-base_lr_10/eval_rrr', 'rb') as file:
    answers = pickle.load(file)

wrong_list = []

with open('./dev.jsonl' ,'r') as reader:
    for line_id,line in enumerate(reader.readlines()):
        t = json.loads(line)
        if answers[line_id] != t['label']:
            t['wrong_label'] = answers[line_id]
            wrong_list.append(t)


with open('wrong.json', 'w') as writer:
    for w in wrong_list:
        writer.writelines(json.dumps(w) + '\n')