import os
import pandas as pd
import csv
import pickle
import numpy as np
import torch

import argparse

def write_answer_to_file(answer, args):

    if not os.path.exists(args.output): os.mkdir(args.output)
    file_path = os.path.join(args.output, "subtask1.csv")
    # turn to Int
    answer = answer.astype(int)
    b = pd.DataFrame(answer, columns=['a']).astype(int)
    b.to_csv(file_path, header=0)
    # import IPython; IPython.embed(); exit(1)
    # np.savetxt(file_path, answer, delimiter=",")


def get_answer(args):
    with open(os.path.join(args.input,"result.pkl"), "rb") as file:
        answer = pickle.load(file)
    return answer

def enssmble(answer):
    weight = [(a["acc"] - 0.8)*100. for a in answer]

    real_answer = np.zeros_like(answer[0]["answer"])
    for idx, w in enumerate(weight):
        real_answer += w*answer[idx]["answer"]
    
    return torch.argmax(torch.tensor(real_answer), dim=1).cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='convert pickle file to csv file')
    parser.add_argument('--input', type=str, help='pickle file path', default="./answer_file/roberta_enhanced")
    parser.add_argument('--output', type=str, help='csv file path', default="./answer_file")
    args = parser.parse_args()

    answer_list = ["./answer_file/roberta_enhanced", "./answer_file/roberta_smooth_label_85"]
    answer = []
    for a in answer_list:
        args.input = a
        answer.append(get_answer(args))

    answer = enssmble(answer)
    write_answer_to_file(answer, args)

    # import IPython; IPython.embed(); exit(1)

    # assert answer["acc"] > 0.82, "什么臭鱼烂虾， 82都不到, guna"
    # answer = torch.argmax(torch.tensor(answer["answer"]), dim=1).cpu().numpy()
    # write_answer_to_file(answer, args)

    
    