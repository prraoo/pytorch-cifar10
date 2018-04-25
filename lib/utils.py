import matplotlib.pyplot as plt
import numpy as np

'''
utility functions for pytorch
    -serializing data
    -data augemntation
    -progress bar
    -visualizations
'''
# addtional test data https://github.com/hardikvasa/google-images-download
# model:  https://github.com/pytorch/pytorch/issues/2001

def imshow(trainloader, classes):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

