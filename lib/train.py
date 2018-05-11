"""
training code for model
"""
import os

import torch
import torch.nn as nn

from lib import utils, parse_config


import torch.optim as optim
from torch.autograd import Variable

from tensorboardX import SummaryWriter

args = parse_config.parser.parse_args()
writer = SummaryWriter(args.expt_name)
embeddings_log = 5

# Training
def train(epoch,trainloader,net,use_cuda, learning_rate):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    #optimizer = optim.Adam(net.parameters(), lr = learning_rate, weight_decay=5e-4)
    for batch_idx, samples in enumerate(trainloader):
        n_iter = (epoch*len(trainloader))+batch_idx
        inputs = samples[0]
        targets = samples[1]
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        #reset grad
        net.zero_grad()
        optimizer.zero_grad()
        # get data batch
        inputs = Variable(inputs, requires_grad=True).float()
        targets =  Variable(targets, requires_grad=False).long()

        #forward
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        #backward
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()
        #logging

        writer.add_scalars("data/scalars_group", {"tr_loss":(train_loss/(batch_idx+1))},epoch)
        writer.add_scalars("data/scalars_group", {"lr":(learning_rate)},epoch)

        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Tr_Acc: %.3f%% (%d/%d)|lr: %.5f'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, learning_rate))



