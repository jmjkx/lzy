import torch
import torch.nn as nn
import torch.nn.functional as F 
from math import sqrt
from manifoldConvLayer import ManifoldConvLayer
from myMFA import myMFA
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from ProjectionMatrix import ProjectionMatrix, ProjectionMatrix1
from ipdb import set_trace

torch.cuda.set_device(0)


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

    def forward(self, x, *y, train=False):


        #  ============================
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[2])
        if train:
            self.W1 = ProjectionMatrix(x, self.Mw, self.Mb, 256).astype(np.float64)
            self.manifold1 = torch.nn.Linear(self.W1.shape[0], self.W1.shape[1], bias=False).double()
            self.manifold1.data = self.W1.astype(np.float64)
            self.manifold1 = self.manifold1.to('cuda')
        x = self.manifold1(x)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[2])
        
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

    net = Net()
    print(net)