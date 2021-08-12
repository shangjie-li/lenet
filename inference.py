import torch
import torch.nn as nn
import matplotlib.image as imgplt
import matplotlib.pyplot as plt
import numpy as np
import argparse

from model import LeNet
from data import data_test_loader

parser = argparse.ArgumentParser(description='LeNet Evaluation')
parser.add_argument('--weight_path', default='./weights/2021-08-10-19-19.pth',
                    type=str, help='path of weight')
args = parser.parse_args()

save_info = torch.load(args.weight_path)
model = LeNet()
criterion = nn.CrossEntropyLoss()
model.load_state_dict(save_info["model"])
model.eval()

test_loss = 0
correct = 0
total = 0

with torch.no_grad(): # turn off dynamic graph
    for batch_idx, (inputs, targets) in enumerate(data_test_loader):
        # inputs: <class 'torch.Tensor'> torch.Size([1, 1, 32, 32])
        outputs = model(inputs)
        
        # targets: <class 'torch.Tensor'> torch.Size([1]) means label (between 0 and 9)
        # outputs: <class 'torch.Tensor'> torch.Size([1, 10]) means confidence of each class
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        
        _, predicted = outputs.max(1) # find max value and indice of dimension 1
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
        
        print(batch_idx, len(data_test_loader),
            'Loss: %.3f | Acc: %.3f%% (%d / %d)' % (
                test_loss / (batch_idx + 1), 100. * correct / total,
                correct, total,
            )
        )
