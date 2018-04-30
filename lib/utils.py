import matplotlib.pyplot as plt
import numpy as np

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
    if epoch == 40:
        lr = args.lr*0.1
    if epoch == 60:
        lr = args.lr*0.01
    return lr
def imshow(trainloader, classes):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 75
#last_time = time.time()
#begin_time = last_time

def progress_bar(current,total,msg=None):
    #global last_time,begin_time
    #if current == 0:
    #   begin_time = time.time()

    curr_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - curr_len)

    sys.stdout.write('[')
    for i in range(curr_len):
        sys.stdout.write("=")
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write(']')

    #curr_time = time.time()
    #step_time = curr_time - last_time
    #last_time = curr_time
    #tot_time = curr_time - begin_time

    L = []
    #L.append(" Step: %s "% format_time(step_time))
    #L.append(" | Tot: %s "% format_time(tot_time))
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

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
