import torch
import torch.nn as nn
import argparse

from model import LeNet
from data import data_train_loader

parser = argparse.ArgumentParser(description='LeNet Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--epoch', default=10, type=int, help='training epoch')
args = parser.parse_args()

model = LeNet()
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
    weight_decay=5e-4,
)

iter_num = 0

train_loss = 0
correct = 0
total = 0

for i in range(args.epoch):
    for batch_idx, (inputs, targets) in enumerate(data_train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        print(
            'epoch:', i,
            'bacth:', batch_idx, '/', len(data_train_loader),
            'Loss: %.3f | Acc: %.3f%% (%d / %d)' % (
                train_loss / (len(data_train_loader) * i + batch_idx + 1),
                100. * correct / total,
                correct, total,
            )
        )
    iter_num += 1

save_info = {
    "iter_num": iter_num,
    "optimizer": optimizer.state_dict(),
    "model": model.state_dict(),
}

torch.save(save_info, './weights/2021-08-10-19-19.pth')
