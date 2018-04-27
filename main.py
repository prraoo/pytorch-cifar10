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


import torch.optim as optim
from torch.autograd import Variable


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

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

#criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader["train"]):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(dataloader["train"]), 'Loss: %.3f | Tr_Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx,(inputs,outputs) in enumerate(dataloader["test"]):
        if use_cuda:
            inputs, outputs = inputs.cuda(), outputs.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(outputs)
        outputs = net(inputs)
        loss = criterion(outputs,targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(dataloader["test"]), "Loss: %.3f | Te_Acc: %.3f (%d,%d)"
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        #save checkpoint
        acc = 100.*correct/total
        print("Saving model..:")
        state={
            "net" : net.module if use_cuda else net,
            "acc" : acc,
            "epoch" : epoch,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        torch.save(state, "./checkpoint/ckpt.t7")
        best_acc = acc

for epoch in range(start_epoch, start_epoch+50):
    train(epoch)
    test(epoch)


############
"""

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    running_loss = 0
    for i, data in enumerate(dataloader["train"],0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i/2000==0:
          print("[%d, %5d] loss:%3f" % (epoch+1, i+1,running_loss/2000))
          running_loss = 0.0
print("Finished Training")

"""


