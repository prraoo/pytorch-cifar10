import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import math
import time

from lib import parse_config

'''
utility functions for pytorch
    -serializing data
    -data augemntation
    -progress bar
    -visualizations
'''
# addtional test data https://github.com/hardikvasa/google-images-download
# visualize model:  https://github.com/pytorch/pytorch/issues/2001

args = parse_config.parser.parse_args()

def lr_multiplier(epoch):
    lr = args.lr
    if epoch >= 40:
        lr = lr*0.1
    if epoch >= 60:
        lr = lr*0.01
    if epoch >= 100:
        lr = lr*0.001
    return lr
def imshow(trainloader, classes):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 75

def progress_bar(current,total,msg=None):

    curr_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - curr_len)

    sys.stdout.write('[')
    for i in range(curr_len):
        sys.stdout.write("=")
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write(']')


    L = []
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width -int(TOTAL_BAR_LENGTH) -len(msg)-3):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current+1,total))
    """
    if current < total-1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    """
    sys.stdout.write("\r")
    sys.stdout.flush()

def plot_confusion(df_confusion, title="confusion matrix",plot_size=(1,2)):

    a = list(plot_size)
    ax=[]
    for i in range(len(a)):
        ax.insert(i,"ax"+str(i))
    ax = tuple(ax)

    fig, ax  = plt.subplots(plot_size)
    for idx,df in enumerate(df_confusion):
        plt.matshow(df[i])
    plt.show()

def confusion(ground_truth,predicted):
    df = {}
    y_actu = pd.Series(ground_truth, name="actual")
    y_pred = pd.Series(predicted, name="predicted")
    df["confusion"] = pd.crosstab(y_actu,y_pred, rownames=['Actual'],colnames=["Predicted"],margins=True)
    df["confusion_norm"] = df["confusion"]/df["confusion"].sum(axis=1)

    return df

