import os
import pandas as pd
import csv
import pickle
import numpy as np
import torch

import argparse

def write_answer_to_file(answer, args):

    if not os.path.exists(args.output): os.mkdir(args.output)
    file_path = os.path.join(args.output, "subtask2.csv")
    # turn to Int
    answer = answer.astype(int)
    b = pd.DataFrame(answer, columns=['a']).astype(int)
    b.to_csv(file_path, header=0)
    # import IPython; IPython.embed(); exit(1)
    # np.savetxt(file_path, answer, delimiter=",")


def get_answer(args):
    with open(args.input, "rb") as file:
        answer = pickle.load(file)
    return answer

def enssmble(answer):
    weight = [(a["acc"] - 0.83)*100. if a["acc"] > 0.8 else 1 for a in answer ]

    real_answer = np.zeros_like(answer[0]["answer"])
    for idx, w in enumerate(weight):
        real_answer += w*answer[idx]["answer"]
    
    return torch.argmax(torch.tensor(real_answer), dim=1).cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='convert pickle file to csv file')
    parser.add_argument('--input', type=str, help='pickle file path', default="./answer_file/roberta_enhanced")
    parser.add_argument('--output', type=str, help='csv file path', default="./answer_file")
    args = parser.parse_args()


    answer = []

    albert_86 = "albert_86_128/result.pkl"
    roberta_sliding_85 = "roberta_sliding-85.5/result.pkl"
    roberta_sliding_87 = "roberta-87-256-smoothing_lr7/result.pkl"
    ronghe = "ronghe/result.pkl"
    robert_90 = "answer_file/task2/roberta_90_1/result.pkl"
    xlnet_86 = "xlnet/result_large_origin.pkl"

    # roberta single model
    # roberta, albert
    # answer_list = ["roberta-87-256-smoothing_lr7/result.pkl", "albert_86_128/result.pkl"]


    # answer_list = [albert_86, roberta_sliding_85, roberta_sliding_87, xlnet_86]

    # # gogogogo
    # answer_list = ["roberta_90/result.pkl", "roberta_90_1/result.pkl"]
    # answer_list = ["roberta/2/86-6.pkl", "roberta/2/89.pkl"]

    """
    5个模型我取最高的,
    """
    answer_list= [
        "albert_decay-89/result.pkl",
        "roberta_90/result.pkl",
        "roberta/2/89.pkl",
        "xlnet/result_large_task2_1.pkl"

    ]




    answer_list = [os.path.join("./answer_file/task2",a) for a in answer_list]

    for a in answer_list:
        args.input = a
        answer.append(get_answer(args))

    answer = enssmble(answer)
    write_answer_to_file(answer, args)

    os.system("zip  -j ./answer_file/1model.zip ./answer_file/subtask2.csv")
    
    # import IPython; IPython.embed(); exit(1)

    # assert answer["acc"] > 0.82, "什么臭鱼烂虾， 82都不到, guna"
    # answer = torch.argmax(torch.tensor(answer["answer"]), dim=1).cpu().numpy()
    # write_answer_to_file(answer, args)

    
    