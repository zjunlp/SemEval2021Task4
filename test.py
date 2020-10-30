from transformers import XLNetTokenizer, XLNetForMultipleChoice
import torch
tokenizer = XLNetTokenizer.from_pretrained('./output/xlnet-base-cased')
model = XLNetForMultipleChoice.from_pretrained('./output/xlnet-base-cased', return_dict=True)
prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
choice0 = "It is eaten with a fork and a knife."
choice1 = "It is eaten while held in the hand."
labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1
encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors='pt', padding=True)
outputs = model(**{k: v.unsqueeze(0) for k,v in encoding.items()}, labels=labels)  # batch size is 1
# the linear classifier still needs to be trained
loss = outputs.loss
logits = outputs.logits
import IPython; IPython.embed(); exit(1)