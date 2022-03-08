import torch
import torch.nn as nn
import torchvision.models as models
from utils import *


class HashingModel(nn.Module):
    def __init__(self, nbit):
        super(HashingModel, self).__init__()

        ''' Img_Net '''
        self.encoder = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            #nn.Tanh(),
            #nn.Dropout(0.3),

            nn.Linear(1000, nbit),
            nn.BatchNorm1d(nbit),
            nn.Tanh()
        )
        

        self.decoder = nn.Sequential(
            nn.Linear(nbit, 500),
            nn.BatchNorm1d(500),
            #nn.ReLU(inplace=True),
            nn.Tanh(),

            nn.Linear(500, 1000),
            nn.BatchNorm1d(1000)
        )

    def forward(self, x):
        code = self.encoder(x)
        feat_ = self.decoder(code)

        return x, code, feat_
