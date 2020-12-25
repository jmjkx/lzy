import os
from datetime import datetime
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ipdb import set_trace

from manifoldConvLayer import ManifoldConvLayer
from myMFA import myMFA
from ProjectionMatrix import ProjectionMatrix, ProjectionMatrix1

torch.cuda.set_device(0)


def build(s: str, loc):
    exec(s, loc)
    return loc
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5).double()
        self.conv2 = nn.Conv2d(6, 6, 5).double()
        #        self.fc1   = nn.Linear(16,10)
        self.fc1 = nn.Linear(384, 10)
        #        self.fc2   = nn.Linear(128, 10)
        
        # self.manifold1 = ManifoldConvLayer(myMFA(1, 144, 10, 20))
        # self.manifold2 = ManifoldConvLayer(myMFA(1, 16, 10, 20))
        self.Mw = np.load('/home/liyuan/Programming/python/lzy/graph/graphs/Mw.npy')
        self.Mb = np.load('/home/liyuan/Programming/python/lzy/graph/graphs/Mb.npy')
    #        self.manifold2 = ManifoldFcLayer(myMFA(1, 10, 3, 5))
    #        self.manifold1 = ManifoldConvLayer(myPCA(144))
    #        self.manifold2 = ManifoldConvLayer(myPCA(16))
    

    def get_manifold(self, x, dim):
        print('===========WWWWWWWWW=================')
        W = ProjectionMatrix(x, self.Mw, self.Mb, dim).astype(np.float64)
        self.manifold = torch.nn.Linear(W.shape[0], W.shape[1], bias=False).double()
        self.manifold.data = W.astype(np.float64)
        self.manifold = self.manifold.to('cuda')
        return self.manifold

    def forward(self, x, *y, train=False):
        self.samplenumber = x.shape[0]
        #  ============================
        x = self.conv1(x) # 6000 6 24 24
        # 建立6通道
        # [x1, x2, x3, x4 ,x5 ,x6]  = map(lambda x, d:x[:, d, :, :].reshape(x.shape[0],\
        #        x.shape[2]*x.shape[2]), [(x, 0), (x, 1), (x, 2), 
        #        (x, 3), (x, 4), (x, 5)])

        x1 = x[:, 0, :, :].reshape(x.shape[0], x.shape[2]*x.shape[2])
        x2 = x[:, 1, :, :].reshape(x.shape[0], x.shape[2]*x.shape[2])
        x3 = x[:, 2, :, :].reshape(x.shape[0], x.shape[2]*x.shape[2])
        x4 = x[:, 3, :, :].reshape(x.shape[0], x.shape[2]*x.shape[2])
        x5 = x[:, 4, :, :].reshape(x.shape[0], x.shape[2]*x.shape[2])
        x6 = x[:, 5, :, :].reshape(x.shape[0], x.shape[2]*x.shape[2])
        

        #分通道流形
        if train:         
            x1 = self.get_manifold(x1, 256)(x1)
            x2 = self.get_manifold(x2, 256)(x2)
            x3 = self.get_manifold(x3, 256)(x3)
            x4 = self.get_manifold(x4, 256)(x4)
            x5 = self.get_manifold(x5, 256)(x5)
            x6 = self.get_manifold(x6, 256)(x6)

        #  变形回去
        x1 = x1.reshape(self.samplenumber, int(x1.shape[0] / self.samplenumber), int(sqrt(x1.shape[1])), \
            int(sqrt(x1.shape[1])))
        x2 = x2.reshape(self.samplenumber, int(x2.shape[0] / self.samplenumber), int(sqrt(x2.shape[1])), \
            int(sqrt(x2.shape[1])))
        x3 = x3.reshape(self.samplenumber, int(x3.shape[0] / self.samplenumber), int(sqrt(x3.shape[1])), \
            int(sqrt(x3.shape[1])))
        x4 = x4.reshape(self.samplenumber, int(x4.shape[0] / self.samplenumber), int(sqrt(x4.shape[1])), \
            int(sqrt(x4.shape[1])))
        x5 = x5.reshape(self.samplenumber, int(x5.shape[0] / self.samplenumber), int(sqrt(x5.shape[1])), \
            int(sqrt(x5.shape[1])))
        x6 = x6.reshape(self.samplenumber, int(x6.shape[0] / self.samplenumber), int(sqrt(x6.shape[1])), \
            int(sqrt(x6.shape[1])))
        x = torch.cat([x1, x2, x3, x4, x5, x6], dim=1, out=None)
        #  ============================


        
        x = self.conv2(x)  
        
        
        x1 = x[:, 0, :, :].reshape(x.shape[0], x.shape[2]*x.shape[2])
        x2 = x[:, 1, :, :].reshape(x.shape[0], x.shape[2]*x.shape[2])
        x3 = x[:, 2, :, :].reshape(x.shape[0], x.shape[2]*x.shape[2])
        x4 = x[:, 3, :, :].reshape(x.shape[0], x.shape[2]*x.shape[2])
        x5 = x[:, 4, :, :].reshape(x.shape[0], x.shape[2]*x.shape[2])
        x6 = x[:, 5, :, :].reshape(x.shape[0], x.shape[2]*x.shape[2])
            
        
        if train:
            x1 = self.get_manifold(x1, 64)(x1)
            x2 = self.get_manifold(x2, 64)(x2)
            x3 = self.get_manifold(x3, 64)(x3)
            x4 = self.get_manifold(x4, 64)(x4)
            x5 = self.get_manifold(x5, 64)(x5)
            x6 = self.get_manifold(x6, 64)(x6)        
            x = torch.cat([x1, x2, x3, x4, x5, x6], dim=1, out=None)
        #    ===================================
        #        set_trace()
        # ===================================
        x = x.view(x.size()[0], -1)  # 展开成一维的

        #        x = F.relu(x)

        x = F.relu(self.fc1(x))
        #       set_trace()
        #        x = F.relu(self.fc2(x))

        #        x = torch.Tensor(x).requires_grad_()
        return x


if __name__ == '__main__':
    print('start')
    net = Net()
    # print(net)
