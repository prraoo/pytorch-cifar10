"""
testing code for model
"""
import torch
import torchvision.utils as vutils

import torch.nn as nn
from lib import utils, parse_config, data_loader


import torch.optim as optim
from torch.autograd import Variable

from tensorboardX import SummaryWriter

args = parse_config.parser.parse_args()
writer = SummaryWriter(args.expt_name)
embeddings_log = 5
best_acc = 0

def test(epoch,testloader,net,use_cuda, learning_rate):
    import shutil
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    def save_checkpoint(state,is_best, filename="./checkpoint/checkpoint.pth.tar"):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, "best_model.pth.tar")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    with torch.no_grad():
        for batch_idx,samples in enumerate(testloader):

            inputs = samples[0]
            targets = samples[1]

            if use_cuda:
                inputs, outputs = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(outputs)

            outputs = net(inputs)
            loss = criterion(outputs,targets)

            test_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            #correct += predicted.eq(targets.data).cpu().sum().item()
            correct += (predicted==targets.data).sum().item()

            Te_Acc = 100.*correct/total
            utils.progress_bar(batch_idx, len(testloader), "Loss: %.3f | Te_Acc: %.3f (%d,%d)"
                    % (test_loss/(batch_idx+1),100.*correct/total, correct, total))

            writer.add_scalars("data/scalars_group", {"te_loss":(test_loss/(batch_idx+1))},epoch)
            writer.add_scalars("data/scalars_group", {"te_acc":(Te_Acc/(batch_idx+1))},epoch)

            #save checkpoint
            is_best = Te_Acc > best_acc
            best_acc = max(Te_Acc, best_acc)
            save_checkpoint({
                "epoch" : epoch+1,
                "args" : args,
                "net" : net.module if use_cuda else net,
                "best_acc" : best_acc,
                "optimizer" : optimizer.state_dict()
                }, is_best)



        print("Saving model..:")
        classes = data_loader.create_class()

        dataiter = iter(testloader)
        img, lbl = dataiter.next()
        _out = net(Variable(img))
        _, predicted = torch.max(_out,1)


        #GroundTruth
        print('GroundTruth: ', ' '.join('%9s' % classes[lbl[j]] for j in range(9)))
        #Predicted
        print('Predicted: ', ' '.join('%9s' % classes[predicted[j]]for j in range(9)))

        #Embeddings
        lbl = [classes[i] for i in lbl]
        writer.add_embedding(_out.data,metadata=lbl,label_img=img.data, global_step=epoch)

        # plot
        df = utils.confusion(targets,predicted)
        print("---------------Printing Confusion Matirx---------------")
        print(df["confusion"])



