import os
from datetime import datetime
from math import sqrt

import numpy as np
import scipy.sparse.linalg as sla
import torch
import torch.nn as nn
import torch.nn.functional as F

from manifoldConvLayer import ManifoldConvLayer
from myMFA import myMFA

torch.cuda.set_device(8)


class SingleChannelConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.Mw = np.load('/home/liyuan/Programming/python/lzy/graph/graphs/Mw.npy')
        self.Mb = np.load('/home/liyuan/Programming/python/lzy/graph/graphs/Mb.npy')
       
    def forward(self, x: torch.tensor, trainable = False, *manifoldlayer: torch.nn.modules.linear.Linear)-> torch.tensor:
        x = x.reshape(x.shape[0], x.shape[2]*x.shape[2])
        if trainable:
            print('===========WWWWWWWWW=================')
            W = SeparableManifoldLayer.ProjectionMatrix(x, self.Mw, self.Mb,144).astype(np.float64)
            self.manifold = torch.nn.Linear(W.shape[0], W.shape[1], bias=False).double()
            self.manifold.data = torch.from_numpy(W.astype(np.float64))
            self.manifold = self.manifold.to('cuda')
            # self.W = nn.Parameter(torch.from_numpy(W).to('cuda'), requires_grad=True)
            
        # else:
        #     self.manifold = manifoldlayer[0]
        # x = torch.mm(x, self.W)
        x = self.manifold(x)
        x = x.reshape(x.shape[0], int(sqrt(x.shape[1])), \
            int(sqrt(x.shape[1])))
        return x


class SeparableManifoldLayer(nn.Module):
    def __init__(self, dim: int)-> None:
        '''
        动态可分离流型层
        '''
        super().__init__()
        self.dim = dim
        # self.params = nn.ParameterList([])
        self.params = []
        self.Mw = np.load('/home/liyuan/Programming/python/lzy/graph/graphs/Mw.npy')
        self.Mb = np.load('/home/liyuan/Programming/python/lzy/graph/graphs/Mb.npy')

        
    @staticmethod
    def ProjectionMatrix(deepFeaTrn: torch.tensor, Mw: np.array, Mb: np.array, dim: int)-> np.array:
        # deepFeaTrn = deepFeaTrn.cpu().clone().detach().numpy()
        # print('======================matmul==============================')
        # A = np.matmul(np.matmul(deepFeaTrn.T, Mb), deepFeaTrn)
        # B = np.matmul(np.matmul(deepFeaTrn.T, Mw), deepFeaTrn)
        # print('======================shape==============================')
        # a = A.shape[0]
        # b = B.shape[0]
        # print('======================eye==============================')
        # A = A + np.eye(a) * 0.01;
        # B = B + np.eye(b) * 0.01;
        # print('======================TTTTTTTTTTT==============================')
        # A = (A + A.T) / 2;
        # B = (B + B.T) / 2;
        # # np.save('A.npy', A)
        # # np.save('B.npy', B)
        # print('eig==============================')
        A = np.load('A.npy')
        B = np.load('B.npy')
        eigvalue, eigvector = sla.eigs(A, dim, B, which='LR')
        return np.real(eigvector)

   
    def forward(self, x:torch.tensor, trainable=False):
        self.trainable = trainable
        """
        按通道分别卷积
        """
        if trainable:
            # self.params.append(SingleChannelConv())
            # x = [Singlechannelconv(x[:, i, :, :]).unsqueeze(1) for i in range(x.shape[1])]
            # x_sep = [Singlechannelconv(x[:, i, :, :]).unsqueeze(1) for i in range(x.shape[1])]
            self.singlechannelconv1 = SingleChannelConv()
            self.singlechannelconv2 = SingleChannelConv()
            x_sep = [self.singlechannelconv1(x[:, 0, :, :], trainable).unsqueeze(1), self.singlechannelconv1(x[:, 1, :, :], trainable).unsqueeze(1)]
            
        else:
            # x_sep = [self.singlechannelconv(x[:, i, :, :], self.params[i]).unsqueeze(1) for i in range(x.shape[1])]
            x_sep = [self.singlechannelconv1(x[:, 0, :, :]).unsqueeze(1), self.singlechannelconv1(x[:, 1, :, :]).unsqueeze(1)]
        x = torch.cat(tuple(x_sep), dim=1)
        return x



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 5).double()
        self.conv2 = nn.Conv2d(2, 2, 5).double()
        self.fc1 = nn.Linear(288, 10)
        self.SeparableManifoldLayer = SeparableManifoldLayer(144).double()

    def forward(self, x, *y, trainable=False):

        self.samplenumber = x.shape[0]
        #  ============================
        x = self.conv1(x) # 6000 6 24 24
        x = self.SeparableManifoldLayer(x, trainable=trainable)
      
        #  ============================
        # x = self.conv2(x)  
           
        # x = self.manifold2(x)
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
    net = Net().double().to('cuda')
    print('==================build===================')
    x = torch.ones(60000, 6, 256, 256).double().to('cuda')
    print('============out==============')
    output = net(x, trainable=True)
    print(net)
