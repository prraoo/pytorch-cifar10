"""
train and test CIFAR10 with pytorch
"""
import os

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision import datasets, transforms

from models import lenet
from lib import utils, data_loader, parse_config

args = parse_config.parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0
start_epoch = 0

transforms = data_loader.create_tr_te_transfrom()
dataloader, _, _ = data_loader.create_tr_te_data(False, transforms["train"], transforms["test"])

classes = data_loader.create_class()

if args.resume:
    #load model from last check point
    print("resuming from last checkpoint...")
    assert os.path.isdir('checkpoint'), "Error: no checkpoint found"
    checkpoint = torch.load("./checkpoint/ckpt.t7")
    net = checkpoint["net"]
    best_acc =  checkpoint["acc"]
    start_epoch =  checkpoint["epoch"]
else:
    print("Building model")
    net = lenet.lenet()
"""
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
"""
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    running_loss = 0
    for i, data in enumerate(dataloader["train"],0):
        inputs, labels = data

        optimizer.zero_grad()

        output = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i/2000 == 1999:
            print("[%d, %5d] loss:%3f"
                    (epoch+1, i+1,running_loss/2000))
            running_loss = 0

print("Finished Training")




