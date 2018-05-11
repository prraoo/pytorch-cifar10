import os
import argparse

import torch
import torchvision
from torchvision import datasets, transforms

data_folder = "data/"

def _get_file_path(filename=""):
    return os.path.join(data_folder,"cifar-10-batches-py",filename)

def _unpickle(filename):
    import pickle
    filepath = _get_file_path(filename)
    print("loading " + filepath)
    with open(filepath, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        return dict

def create_tr_te_transfrom():
    print("transforming data ..")
    transform_train = transforms.Compose([
                      transforms.RandomCrop(32, padding=4),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize((0.49139968, 0.48215841 ,0.44653091), (0.24703223 ,0.24348513 ,0.26158784)),
                      ])

    transform_test = transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.49139968 ,0.48215841 ,0.44653091), (0.24703223 ,0.24348513 ,0.26158784)),
                     ])


    trnfm = {}
    trnfm["train"] = transform_train
    trnfm["test"] = transform_test

    return trnfm



def create_tr_te_data(download="False", transform_tr=None, transform_te=None):

    tr_data = torchvision.datasets.CIFAR10(data_folder, train=True,
            transform=transform_tr, download=download)
    trainloader = torch.utils.data.DataLoader(tr_data,
            batch_size=32, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(data_folder, train=False,
            transform=transform_te, download=download)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    return trainloader, testloader

def create_class():
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return classes



#### scratch pad ####
if __name__ == "__main__":
    trfrm = create_tr_te_transfrom()
    _data_set,_tr,_te = create_tr_te_data()

    print(_data_set["test"][1])

    #print(_tr.train_data.mean(axis=(0,1,2))/255)
    #print(_tr.train_data.std(axis=(0,1,2))/255)

    #mean = [0.49139968 0.48215841 0.44653091]
    #std = [0.24703223 0.24348513 0.26158784]
