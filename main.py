import numpy as np
import random
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from transformers import ElectraForSequenceClassification, AdamW, BertConfig

from SNS_Dataset import SNS_Dataset


def fix_seeds(seed = 1017):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(device = 'cuda'):
    task = "m0.5_l8"
    mixup = 0.5
    wrs = True
    save_path = "/data/jinu/pt_files/study"

    print(" -- Create Dataloader")
    train_set = SNS_Dataset(set_type='train', mixup=mixup)
    valid_set = SNS_Dataset(set_type='valid')

    classes = None

    if(wrs):
        class_count = [44.87, 9.08, 14.94, 45.46, 4.27, 1.72, 5.15, 4.27]
        class_weights = [1/i for i in class_count]
        weights = [class_weights[train_set.__getlabel__(i)] for i in range(len(train_set))]

        sampler = torch.utils.data.WeightedRandomSampler(weights, len(train_set), replacement=True)
        train_loader = DataLoader(train_set, batch_size=384, sampler=sampler)
    else:
        train_loader = DataLoader(train_set, batch_size=384, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=384, shuffle=True)

    print(" -- Create Model")
    model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator", num_labels=8)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-8)

    min_loss = 100000
    print(" -- Training")
    for epoch in range(1000):
        model.train()
        loss_sum = 0
        acc_sum = 0
        for batch, (input_ids, one_hot_labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()
            input_ids = input_ids.to(device)
            one_hot_labels = one_hot_labels.to(device)

            outputs = model(input_ids, labels=one_hot_labels)

            logits = outputs.logits
            logits = torch.nn.functional.softmax(logits, dim=-1)

            loss = one_hot_labels*(-torch.log(logits+1e-10))
            loss = loss.sum(dim=-1).mean()

            acc = 0
            for i in range(len(input_ids)):
                if(logits[i].argmax() in one_hot_labels[i].nonzero()):
                    acc += 1
            acc /= len(input_ids)

            loss.backward()
            loss_sum += loss.item()
            acc_sum += acc
            optimizer.step()

            if batch % 100 == 99:
                loss_sum /= 100
                acc_sum /= 100
                print("Epoch {} Batch {} Loss {} acc {}".format(epoch, batch+1, loss_sum, acc_sum))
                loss_sum = 0
                acc_sum = 0


        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_acc = 0
            for batch, (input_ids, one_hot_labels) in tqdm(enumerate(valid_loader)):
                input_ids = input_ids.to(device)
                one_hot_labels = one_hot_labels.to(device)
                
                outputs = model(input_ids, labels=one_hot_labels)
                
                logits = outputs.logits
                logits = torch.nn.functional.softmax(logits, dim=-1)

                loss = one_hot_labels*(-torch.log(logits+1e-10))
                loss = loss.sum(dim=-1).mean()

                acc = 0
                for i in range(len(input_ids)):
                    if(logits[i].argmax() in one_hot_labels[i].nonzero()):
                        acc += 1
                acc /= len(input_ids)

                total_loss += loss.item()*len(input_ids)
                total_acc += acc*len(input_ids)
            
            total_loss /= len(valid_set)
            total_acc /= len(valid_set)

            print("Valid Epoch {} Loss {} acc {}".format(epoch, total_loss, total_acc))

            if(total_loss < min_loss):
                min_loss = total_loss
                torch.save(model.state_dict(), f"{save_path}/{task}_{epoch}_{round(total_loss,4)}+{round(total_acc,4)}.ptl")
                print(f"{save_path}/{task}_{epoch}_{round(total_loss,4)}+{round(total_acc,4)}.pt")


def main():
    fix_seeds()
    train()

main()