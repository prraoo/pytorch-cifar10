import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def conv3x3(inplanes,outplanes,stride=1):
    """3x3 conv with padding"""
    return nn.Conv2d(inplanes,outplanes,kernel_size=3,stride=stride,padding=1, bias=False)

class BasicBlock(nn.Module):
    """
    expansion used in increasing the channels after 1x1 conv
    """

    expansion = 1

    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        #ToDo check ReLU inline
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes,planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.shortcut = nn.Sequential()
        if self.stride !=1 or inplanes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(inplanes,self.expansion*planes,kernel_size=1,stride=self.stride,bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                    )

    def forward(self,x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return(out)

"""
class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleNeck,self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1)
        self.bn3 = nn.Conv2d(planes*4)
        self.relu = nn.ReLU()

        self.downsample = downsample
        self.stride = stride

        self.shortcut = nn.Sequential()
        if self.stride !=1 or inplanes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(inplanes,self.expansion*planes,kernel_size=1,stride=stride,bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                    )

    def forward(self,x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out+= self.shortcut(out)

        out = self.relu(out)


        return out
"""

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        #ToDo - vary different plane
        super(ResNet,self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0],stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1],stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2],stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3],stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        strides = [stride]+[1]*(num_blocks-1)
        for stride in strides:
            layers.append(block(self.inplanes,planes,stride))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x,4)
        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return x

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

if __name__ == "__main__":
    net = ResNet18()
    y = net(Variable(torch.randn(128,3,32,32)))
    print(y.size())
