import os
import shutil

from typing import Dict, Optional
from transformers import EvalPrediction,
import numpy as np 


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



if __name__ == "__main__":
    delete_checkpoint_files_except_the_best("output/roberta-large_enhanced_label_smoothing_task1_testac")
    pass
    