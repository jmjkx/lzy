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
        self.conv2 = nn.Conv2d(1, 1, 5).double()
        #        self.fc1   = nn.Linear(16,10)
        self.fc1 = nn.Linear(144, 10)
        #        self.fc2   = nn.Linear(128, 10)
           
        # self.manifold1 = ManifoldConvLayer(myMFA(1, 144, 10, 20))
        # self.manifold2 = ManifoldConvLayer(myMFA(1, 16, 10, 20))
        self.Mw = np.load('/home/liyuan/Programming/python/lzy/graph/graphs/Mw.npy')
        self.Mb = np.load('/home/liyuan/Programming/python/lzy/graph/graphs/Mb.npy')
    #        self.manifold2 = ManifoldFcLayer(myMFA(1, 10, 3, 5))
    #        self.manifold1 = ManifoldConvLayer(myPCA(144))
    #        self.manifold2 = ManifoldConvLayer(myPCA(16))
    

    def get_manifold(self, x):
        print('===========WWWWWWWWW=================')
        W = ProjectionMatrix(x, self.Mw, self.Mb, 256).astype(np.float64)
        manifold = torch.nn.Linear(W.shape[0], W.shape[1], bias=False).double()
        manifold.data = W.astype(np.float64)
        manifold = manifold.to('cuda')
        return manifold

    def forward(self, x, *y, train=False):
        self.samplenumber = x.shape[0]
        #  ============================
        x = self.conv1(x) # 6000 6 24 24
        # x = x.reshape(x.shape[0]*x.shape[1], x.shape[2]*x.shape[2])

        
            # L = build('x%s = x[:, %s, :, :]'%(i, i), locals())
            # locals().update(L)
        # for i in range(x.shape[1]):
        #         LL = build(s1%(i+1, i), locals().copy())
        #         locals().update(LL)
        # s1 = 'x%s = x[:, %s, :, :].reshape(x.shape[0], x.shape[2]*x.shape[2])'
        # s2 = 'self.manifold%s = self.get_manifold(x%s)\nx%s = self.manifold%s(x%s)'
        # s3 = 'x%s = x%s.reshape(self.samplenumber, int(x%s.shape[0] / self.samplenumber), int(sqrt(x%s.shape[1])), \
        #     int(sqrt(x%s.shape[1])))'

        for i in range(x.shape[1]):
            exec(s1%(i, i))
        # x1 = x[:, 0, :, :].reshape(x.shape[0], x.shape[2]*x.shape[2])
        # x2 = x[:, 1, :, :].reshape(x.shape[0], x.shape[2]*x.shape[2])
        # x3 = x[:, 2, :, :].reshape(x.shape[0], x.shape[2]*x.shape[2])
        # x4 = x[:, 3, :, :].reshape(x.shape[0], x.shape[2]*x.shape[2])
        # x5 = x[:, 4, :, :].reshape(x.shape[0], x.shape[2]*x.shape[2])
        # x6 = x[:, 5, :, :].reshape(x.shape[0], x.shape[2]*x.shape[2])
        
        # 建立6通道
        

        if train:         
            
        #  流通道分流
            for i in range(x.shape[1]):
                L = build(s2%(i+1, i+1, i+1, i+1, i+1), locals())
                locals().update(L)
        #  变形回去
            # for i in range(x.shape[1]):
            #     L = build(s3%(i+1, i+1, i+1, i+1, i+1), locals())
            #     locals().update(L)
        # x1 = x1.reshape(self.samplenumber, int(x1.shape[0] / self.samplenumber), int(sqrt(x1.shape[1])), \
        #     int(sqrt(x1.shape[1])))

        # x2 = x2.reshape(self.samplenumber, int(x2.shape[0] / self.samplenumber), int(sqrt(x2.shape[1])), \
        #     int(sqrt(x2.shape[1])))

        # x3 = x3.reshape(self.samplenumber, int(x3.shape[0] / self.samplenumber), int(sqrt(x3.shape[1])), \
        #     int(sqrt(x3.shape[1])))

        # x4 = x4.reshape(self.samplenumber, int(x4.shape[0] / self.samplenumber), int(sqrt(x4.shape[1])), \
        #     int(sqrt(x4.shape[1])))

        # x5 = x5.reshape(self.samplenumber, int(x5.shape[0] / self.samplenumber), int(sqrt(x5.shape[1])), \
        #     int(sqrt(x5.shape[1])))

        # x6 = x6.reshape(self.samplenumber, int(x6.shape[0] / self.samplenumber), int(sqrt(x6.shape[1])), \
        #     int(sqrt(x6.shape[1])))
       

        #  ============================
        x = self.conv2(x)  
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[2])
        if train:
            self.W2 = ProjectionMatrix1(x, self.Mw, self.Mb, 144).astype(np.float64)
            self.manifold2 = torch.nn.Linear(self.W2.shape[0], self.W2.shape[1], bias=False).double()
            self.manifold2 = self.manifold2.to('cuda')
            self.manifold2.data = self.W2.astype(np.float64)        
        x = self.manifold2(x)
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
