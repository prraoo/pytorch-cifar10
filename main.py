"""
train and test CIFAR10 with pytorch
"""
import os

import torch
import torchvision
from torchvision import datasets, transforms

from models import *
from lib import utils, data_loader, parse_config

args = parse_config.parser.parse_args()


