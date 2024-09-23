import torch

def fix_mono(dataset):
    for data in dataset:
        if data[0].shape[0] == 1:
            torch.cat(data[0], data[0])