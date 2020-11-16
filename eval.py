import torch
import transformers 


with open('./saved_model_file/roberta-task2/training_args.bin','rb') as file:
     a = torch.load(file)

     import IPython; IPython.embed(); exit(1)