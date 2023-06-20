import torch
import random
import pickle
from torch.utils.data import Dataset
from transformers import ElectraTokenizer


class SNS_Dataset(Dataset):
    def __init__(self, set_type, mixup=0):
        self.set_type = set_type

        # load data from pickle file
        self.data = []
        self.label = []
        if self.set_type == 'train':
            with open("/data/jinu/raw_data/Korean_SNS/Bad_Word/final_trainingData.pkl", 'rb') as dataf:
                self.data = pickle.load(dataf)
            with open("/data/jinu/raw_data/Korean_SNS/Bad_Word/final_trainingLabel.pkl", 'rb') as labelf:
                self.label = pickle.load(labelf)
        elif self.set_type == 'valid':
            with open("/data/jinu/raw_data/Korean_SNS/Bad_Word/final_validationData.pkl", 'rb') as dataf:
                self.data = pickle.load(dataf)
            with open("/data/jinu/raw_data/Korean_SNS/Bad_Word/final_validationLabel.pkl", 'rb') as labelf:
                self.label = pickle.load(labelf)
        

        self.max_len = 32
        self.tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.mixup = mixup
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.label[idx]

        # mixup 2 data if mixup > random.random()
        if(self.mixup > random.random()):
            token = self.tokenizer.tokenize(text)

            if(len(token) > int((self.max_len-2)/2)):
                start = random.randint(0, int((self.max_len-2)/2))
                token = token[start:start+int((self.max_len-2)/2)]
            else:
                token = token[:int((self.max_len-2)/2)]

            token_ids = self.tokenizer.convert_tokens_to_ids(token)

            label_detail = label[1]
            # label = label[0]

            one_hot_labels = torch.zeros((8))
            for l in label_detail:
                one_hot_labels[l] = 1
            one_hot_labels /= one_hot_labels.sum()

            if(0 in label_detail):
                while(True):
                    rand_idx = random.randint(0, len(self.data)-1)
                    if(0 in self.label[rand_idx][1]):
                        break
            else:
                while(True):
                    rand_idx = random.randint(0, len(self.data)-1)
                    if(0 not in self.label[rand_idx][1]):
                        break
    
            rand_text = self.data[rand_idx]
            rand_label = self.label[rand_idx]
            rand_token = self.tokenizer.tokenize(rand_text)

            if(len(rand_token) > int((self.max_len-2)/2)):
                start = random.randint(0, int((self.max_len-2)/2))
                rand_token = rand_token[start:start+int((self.max_len-2)/2)]
            else:
                rand_token = rand_token[:int((self.max_len-2)/2)]

            rand_token_ids = self.tokenizer.convert_tokens_to_ids(rand_token)

            rand_label_detail = rand_label[1]

            rand_one_hot_labels = torch.zeros((8))
            for l in rand_label_detail:
                rand_one_hot_labels[l] = 1
            rand_one_hot_labels /= rand_one_hot_labels.sum()

            token_ids = [2] + token_ids + rand_token_ids + [3]

            one_hot_labels = (one_hot_labels + rand_one_hot_labels) / (one_hot_labels + rand_one_hot_labels).sum()
        # otherwise just return one data
        else:
            token = self.tokenizer.tokenize(text)

            if(len(token) > self.max_len-2):
                start = random.randint(0, len(token)-self.max_len+2)
                token = token[start:start+self.max_len-2]
            else:
                token = token[:self.max_len-2]

            token_ids = self.tokenizer.convert_tokens_to_ids(token)
            token_ids = [2] + token_ids + [3]

            label_detail = label[1]

            one_hot_labels = torch.zeros((8))
            for l in label_detail:
                one_hot_labels[l] = 1
            one_hot_labels /= one_hot_labels.sum()

        
        token_ids += [0] * (self.max_len - len(token_ids))

        return torch.tensor(token_ids), one_hot_labels

    def __getlabel__(self, idx):
        label = self.label[idx][1]
        return random.sample(label, 1)[0]