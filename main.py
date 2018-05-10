"""
train and test CIFAR10 with pytorch
"""
import os

import torch
import torch.backends.cudnn as cudnn

from models import lenet, resnet
from lib import data_loader, parse_config, utils
from lib.train import train
from lib.test import test

from tensorboardX import SummaryWriter

args = parse_config.parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0
start_epoch = 0
writer = SummaryWriter(args.expt_name)
embeddings_log = 5

# for old GPUs
use_cuda = True

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
    #net = resnet.ResNet18()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

if __name__ == "__main__":

    trainloader = dataloader["train"]
    testloader = dataloader["test"]

    print("USE CUDA : ", use_cuda)
    for epoch in range(start_epoch, start_epoch+2):
        lr = utils.lr_multiplier(epoch)
        train(epoch=epoch, trainloader=trainloader, net=net, use_cuda=use_cuda,learning_rate=lr)
        test(epoch, testloader=testloader, net=net, use_cuda=use_cuda,learning_rate= lr)

    #writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
