import argparse

parser = argparse.ArgumentParser(description= "Pytorch Training on CIFAR10")
parser.add_argument("--lr", default = 0.01, type=float,help="learning rate")
parser.add_argument("--resume", action="store_true",help="resume from last checkpoint")
parser.add_argument("--expt_name",default="test_dir", required=False,type=str, help="tensorboard experiment names")
parser.add_argument("--gpu", default = True, type=bool,help="self false for incompatible cuda/ old gpus")
parser.add_argument("--download", default = False, type=bool,help="download CIFAR-10 dataset ")
if __name__== "__main__":
    args = parser.parse_args()


