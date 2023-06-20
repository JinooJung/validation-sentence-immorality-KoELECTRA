import numpy as np
import random
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from transformers import ElectraForSequenceClassification, AdamW, BertConfig

from SNS_Dataset import SNS_Dataset

import pickle


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

intensity_keys = [1.0, 1.4, 1.6, 1.8, 2.0]
result_intensity = {key:[] for key in intensity_keys}

valid_set = SNS_Dataset(set_type='valid')


model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator", num_labels=8)

# baseline model - 1
# model.load_state_dict(torch.load("/data/jinu/pt_files/study/save/m0_l7_98_0.948+0.7699.ptl"))

# mixup model - 2
model.load_state_dict(torch.load("/data/jinu/pt_files/study/save/m0.5_l7_98_1.0009+0.7547.ptl"))

# model.to(device)
# model.eval()
# with torch.no_grad():
#     for i in tqdm(range(len(valid_set))):
#         input_ids, label = valid_set[i]
#         input_ids = input_ids.to(device)
#         if(label[0]):
#             continue
#         intensity = valid_set.label[i][2]

#         outputs = model(input_ids.unsqueeze(0))
#         logits = outputs[0]
#         logits = torch.softmax(logits, dim=1)
#         censure_level = torch.sum(logits[0][1:])

#         if(1.0<=intensity<1.4):
#             result_intensity[1.0].append(float(censure_level))
#         elif(1.4<=intensity<1.6):
#             result_intensity[1.4].append(float(censure_level))
#         elif(1.6<=intensity<1.8):
#             result_intensity[1.6].append(float(censure_level))
#         elif(1.8<=intensity<2.0):
#             result_intensity[1.8].append(float(censure_level))
#         elif(2.0<=intensity):
#             result_intensity[2.0].append(float(censure_level))


# with open("result_intensity1.pkl", "wb") as f:
#     pickle.dump(result_intensity, f)
# with open("result_intensity2.pkl", "wb") as f:
#     pickle.dump(result_intensity, f)


with open("result_intensity1.pkl", "rb") as f:
    result_intensity = pickle.load(f)
# with open("result_intensity2.pkl", "rb") as f:
#     result_intensity = pickle.load(f)

import matplotlib.pyplot as plt

print([np.median(result_intensity[key]) for key in intensity_keys])

plt.boxplot([result_intensity[key] for key in intensity_keys])

plt.savefig("result_intensity_1.png")
# plt.savefig("result_intensity_2.png")