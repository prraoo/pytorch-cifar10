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

from tensorboardX import SummaryWriter

args = parse_config.parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0
start_epoch = 0
writer = SummaryWriter()

# for old GPUs
use_cuda = False

transforms = data_loader.create_tr_te_transfrom()
dataloader, _, _ = data_loader.create_tr_te_data(False, transforms["train"], transforms["test"])

classes = data_loader.create_class()

if args.resume:

    #load model from last check point
    print("resuming from last checkpoint...")
    assert os.path.isfile('checkpoint/checkpoint.pth.tar'), "Error: no checkpoint found"
    checkpoint = torch.load("./checkpoint/checkpoint.pth.tar")
    start_epoch =  checkpoint["epoch"]
    net = checkpoint["net"]
    best_acc =  checkpoint["best_acc"]
    optimizer = (checkpoint["optimizer"])
    print("Loaded model from {} , epoch {}".format("./checkpoint/checkpoint.pth.tar", checkpoint["epoch"]))
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
        #tensorboard
        writer.add_scalars("data/scalars_group", {"tr_loss":(train_loss/(batch_idx+1))},epoch)

def test(epoch):
    import shutil
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    def save_checkpoint(state,is_best, filename="./checkpoint/checkpoint.pth.tar"):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, "best_model.pth.tar")

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
        Te_Acc = 100.*correct/total

        utils.progress_bar(batch_idx, len(dataloader["test"]), "Loss: %.3f | Te_Acc: %.3f (%d,%d)"
                % (test_loss/(batch_idx+1), Te_Acc, correct, total))

        #save checkpoint
        is_best = Te_Acc > best_acc
        best_acc = max(Te_Acc, best_acc)
        save_checkpoint({
            "epoch" : epoch+1,
            "args" : args,
            "net" : net.module if use_cuda else net,
            "best_acc" : best_acc,
            "optimizer" : optimizer.state_dict()
            }, is_best)
        writer.add_scalar("Train/loss", Te_Acc, epoch)



    print("Saving model..:")
for epoch in range(start_epoch, start_epoch+1):
    train(epoch)
    test(epoch)

writer.export_scalars_to_json("./all_scalars.json")
writer.close()
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


