import torch.nn as nn
import numpy as np
import torch
from models.model_h import *
from utils import *
import torch.nn.functional as F

def Training(args, loader):

    criterion_l2 = nn.MSELoss().cuda()

    hashing_net = HashingModel(args.nbit)
    hashing_net.cuda()
    optimizer = torch.optim.Adam(hashing_net.parameters(), args.lr)

    for epoch in range(args.num_epoch):

        loss1 = 0.0
        loss2 = 0.0
        loss3 = 0.0
        train_loss = 0.0

        hashing_net.train()

        for step, (img, _, index) in enumerate(loader):

            optimizer.zero_grad()
            img = img.cuda().to(torch.float)
     
            F_I = F.normalize(img)
            S_I = F_I.mm(F_I.t())

            _, h, feat_reconst = hashing_net(img)
            b = torch.sign(h)

            h_norm = F.normalize(h)
            S_h = h_norm.mm(h_norm.t())

            CCRloss = criterion_l2(S_h, S_I) * args.lamda1
            DISloss = criterion_l2(h, b) * args.lamda2
            ACRloss = criterion_l2(feat_reconst, img) * args.lamda3

            # total loss
            loss = CCRloss + DISloss + ACRloss

            loss.backward()
            optimizer.step()

            # show loss
            train_loss += loss.detach().item()
            loss1 += CCRloss.detach().item()
            loss2 += DISloss.detach().item()
            loss3 += ACRloss.detach().item()

        #print('#Epoch %3d: Total Loss: %.8f, loss1: %.8f, loss2: %.8f, loss3: %.8f, ' % (
        #    epoch, args.batchsize*train_loss / len(loader.dataset),
        #    args.batchsize*loss1 / len(loader.dataset),
        #    args.batchsize * loss2 / len(loader.dataset),
        #    args.batchsize * loss3 / len(loader.dataset)
        #))

        # Save models checkpoints
        if (((epoch + 1) % 10) == 0) or (epoch == (args.num_epoch - 1) ):
            if args.dataset == 'coco':
                save_path = 'checkpoints/coco/hashing_net_' + str(args.nbit) + 'bit_epoch' + str(epoch + 1) + '.pth'
            elif args.dataset == 'nus':
                save_path = 'checkpoints/nus/hashing_net_' + str(args.nbit) + 'bit_epoch' + str(epoch + 1) + '.pth'
            elif args.dataset == 'mir':
                save_path = 'checkpoints/mir/hashing_net_' + str(args.nbit) + 'bit_epoch' + str(epoch + 1) + '.pth'
            torch.save(hashing_net, save_path)

    return hashing_net