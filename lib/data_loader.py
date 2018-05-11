import os
import argparse

import torch
import torchvision
from torchvision import datasets, transforms

data_folder = "data/"

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
                     transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.Normalize((0.49139968 ,0.48215841 ,0.44653091), (0.24703223 ,0.24348513 ,0.26158784)),
                     ])


    trnfm = {}
    trnfm["train"] = transform_train
    trnfm["test"] = transform_test

    return trnfm



def create_tr_te_data(download="False", transform_tr=None, transform_te=None):
    data_set = {}
    #train data
    tr_data = torchvision.datasets.CIFAR10(data_folder, train=True,
            transform=transform_tr, download=download)
    trainloader = torch.utils.data.DataLoader(tr_data,
            batch_size=32, shuffle=True, num_workers=2)
    data_set["train"] = trainloader

    #test data
    te_data = torchvision.datasets.CIFAR10(data_folder, train=False,
           target_transform=transform_te, download=download)
    testloader = torch.utils.data.DataLoader(te_data,
           batch_size=32, shuffle=True, num_workers=2)
    data_set["test"] = testloader

    return data_set, tr_data, te_data

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

"""
# get some random training images
dataiter = iter(_data_set["train"])
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# without using pytorch built-in package
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

def _create_tr_te_data(batch_size):
    for i in range(batch_size):
        data = _unpickle("data_batch_"+str(i+1))
    raw_img = data[b'data']
    cls = data[b'labels']
    return data, raw_img, cls
"""
