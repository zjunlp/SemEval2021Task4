import os
import shutil

from typing import Dict, Optional
from transformers import EvalPrediction
import numpy as np 
import pandas as pd


def delete_checkpoint_files_except_the_best(file_path="."):
    """
    已经checkpoint最大的那个是最好的文件，所以我们删去除了最大的那个之外的所有checkpoint文件。
    之后将那个保存到该目录，可以直接from_pretrain()这个目录
    """

    file_list = []
    for root, dirs, files in os.walk(file_path):
        if "checkpoint" in root:
            file_list.append(root)
    
    # import IPython; IPython.embed(); exit(1)
    file_list.sort(key = lambda x: int(x.split('-')[-1]))
    best_file_path = file_list[-1]
    # import IPython; IPython.embed(); exit(1)
    # 删除除了最后一个文件之外的文件
    for file in file_list[:-1]:
        shutil.rmtree(file)

    file_best = os.listdir(best_file_path)
    for f in file_best:
        print(f)
        shutil.move(os.path.join(best_file_path,f), os.path.join(file_path,f))

def simple_accuracy(preds, labels):
    return (preds == labels).mean()



def compute_metrics(p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": simple_accuracy(preds, p.label_ids)}




def write_answer_to_file(answer):
    name = "subtask1.csv" if "task1" in args.data_dir else "subtask2.csv"
    file_path = os.path.join("./answer_file", name)
    # turn to Int
    answer = answer.astype(int)
    b = pd.DataFrame(answer, columns=['a']).astype(int)
    b.to_csv(file_path, header=0)
    # import IPython; IPython.embed(); exit(1)
    # np.savetxt(file_path, answer, delimiter=",")


# def convert_dev_to_train():
#     answer = []
#     with open("./dataset/task1/train.jsonl") as file:
#         for line in file.readlines():
            




if __name__ == "__main__":
    delete_checkpoint_files_except_the_best("output/roberta-large_enhanced_label_smoothing_task1_testac")
    pass
    