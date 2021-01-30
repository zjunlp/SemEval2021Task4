import torch
import transformers 
import argparse
from utils import get_diffience_between_input


parser = argparse.ArgumentParser(description="123")
parser.add_argument("path", type=str, nargs="+", help="")

args = parser.parse_args()

print(get_diffience_between_input(*tuple(args.path)))
