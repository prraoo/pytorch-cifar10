import os
import torch
import torchvision
from torchvision import datasets, transforms

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

#using pytorch built in package
def th_tr_te_transfrom():
    print("transforming data ..")
    transform_train = transforms.Compose([
                      transforms.RandomCrop(32, padding=4),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                      ])

    transform_test = transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                     ])


    trnfm = {}
    trnfm["train"] = transform_train
    trnfm["test"] = transform_test
    
    return trnfm



def th_tr_te_data(download="False", transform_tr=None, transform_te=None):
    data_set = {}
    #train data
    tr_data = torchvision.datasets.CIFAR10(data_folder, train=True,
            transform=transform_tr, download=download)

    trainloader = torch.utils.data.DataLoader(tr_data,
            batch_size=32, shuffle=True, num_workers=2)

    data_set["train"] = trainloader

    #test data
    te_data = torchvision.datasets.CIFAR10(data_folder, train=False,
           target_transform=transfrom_te, download=download)

    testloader = torch.utils.data.DataLoader(te_data,
           batch_size=32, shuffle=True, num_workers=2)

    data_set["test"] = trainloader

    return data_set


####
trfrm = th_tr_te_transfrom()
print(trfrm)
#tr = th_tr_te_data(download="True")
#tr = th_tr_te_data(download="True", transform_tr=transform["train"],transform_te=transfrom["test"])
#print(type(tr))
