import os
from datetime import datetime
from math import sqrt

import numpy as np
import scipy.sparse.linalg as sla
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.function_base import append
from torch.nn.parameter import Parameter

from manifoldConvLayer import ManifoldConvLayer
from myMFA import myMFA

# class SingleChannelConv(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.Mw = np.load('/home/liyuan/Programming/python/lzy/graph/graphs/Mw.npy')
#         self.Mb = np.load('/home/liyuan/Programming/python/lzy/graph/graphs/Mb.npy')
       
#     def forward(self, x: torch.tensor, trainable = False, *manifoldlayer: torch.nn.modules.linear.Linear)-> torch.tensor:
#         x = x.reshape(x.shape[0], x.shape[2]*x.shape[2])
#         if trainable:
#             print('===========WWWWWWWWW=================')
#             W = SeparableManifoldLayer.ProjectionMatrix(x, self.Mw, self.Mb,144).astype(np.float64)
#             self.manifold = torch.nn.Linear(W.shape[0], W.shape[1], bias=False).double()
#             self.manifold.data = torch.from_numpy(W.astype(np.float64))
#             self.manifold = self.manifold.to('cuda')
#             # self.W = nn.Parameter(torch.from_numpy(W).to('cuda'), requires_grad=True)
            
#         # else:
#         #     self.manifold = manifoldlayer[0]
#         # x = torch.mm(x, self.W)
#         x = self.manifold(x)
#         x = x.reshape(x.shape[0], int(sqrt(x.shape[1])), \
#             int(sqrt(x.shape[1])))
#         return x


class SeparableManifoldLayer(nn.Module):
    def __init__(self, ori_dim: int, dim: int, C_dim: int, Mw, Mb)-> None:
        '''
        动态可分离流型层
        oir_dim 输入的维数
        dim 降至维数
        C_dim 通道数，分流
        '''
        super().__init__()
        self.dim = dim
        self.ori_dim = ori_dim
        self.C_dim = C_dim
        self.params = nn.ParameterList([nn.Parameter(torch.randn(self.ori_dim, self.dim).double().to('cuda'), requires_grad=True) for i in range(C_dim)])
        self.Mw = Mw
        self.Mb = Mb 

        
    def ProjectionMatrix(self, deepFeaTrn: torch.tensor, dim)-> np.array:

        # deepFeaTrn = deepFeaTrn.cpu().clone().detach().numpy()
        deepFeaTrn = deepFeaTrn.float()
       
        print('======================matmul==============================')
        deepFeaTrn = deepFeaTrn.to('cuda:1')
        A = torch.mm(torch.mm(deepFeaTrn.T,self.Mb), deepFeaTrn)
        deepFeaTrn = deepFeaTrn.to('cuda:0') 
        B = torch.mm(torch.mm(deepFeaTrn.T,self.Mw), deepFeaTrn)
        a = A.shape[0]
        b = B.shape[0]
        A = A + torch.eye(a).to('cuda:1') * 0.01;
        B = B + torch.eye(b).to('cuda:0') * 0.01;
        A = (A + A.T) / 2;
        B = (B + B.T) / 2;
        print('==============================eig==============================')
        eigvalue, eigvector = sla.eigs(A.cpu().clone().detach().numpy(), dim, \
            B.cpu().clone().detach().numpy(), which='LR')
        print('========================eigend=================================')
        return np.real(eigvector)

   
    def forward(self, x:torch.tensor, trainable=False)-> None:
        """
        按通道分别卷积
        """
        if trainable:
            for i in range(x.shape[1]):
                W = self.ProjectionMatrix(x[:, i, :, :].view(x.shape[0], x.shape[2]*x.shape[2]), self.dim)
                self.params[i].data = torch.from_numpy(W).double().cuda()
        #         self.params.append(nn.Parameter(torch.rand(576, 256).double().to('cuda'), requires_grad=True))
        
        x_sep = [torch.mm(x[:, i, :, :].view(x.shape[0], x.shape[2]*x.shape[2]), self.params[i]) for i in range(x.shape[1])]
        x = torch.cat(tuple(x_sep), dim=1)
        x = x.view(x.shape[0], self.C_dim, int(sqrt(x.shape[1]/self.C_dim)), int(sqrt(x.shape[1]/self.C_dim)))

        # else:
        #     # x_sep = [self.singlechannelconv(x[:, i, :, :], self.params[i]).unsqueeze(1) for i in range(x.shape[1])]
        #     x_sep = [self.singlechannelconv1(x[:, 0, :, :]).unsqueeze(1), self.singlechannelconv1(x[:, 1, :, :]).unsqueeze(1)]
        # x = torch.cat(tuple(x_sep), dim=1) 
        return x



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 3, 5)
        self.fc1 = nn.Linear(192, 10)
        self.Mw = np.load('/home/liyuan/Programming/python/lzy/graph/graphs/Mw.npy')
        self.Mw = torch.from_numpy(self.Mw).to('cuda:0') 
        self.Mb = np.load('/home/liyuan/Programming/python/lzy/graph/graphs/Mb.npy')
        self.Mb = torch.from_numpy(self.Mb).to('cuda:1')
        self.SeparableManifoldLayer2 = SeparableManifoldLayer(144, 64, 3, self.Mw, self.Mb)
        self.SeparableManifoldLayer = SeparableManifoldLayer(576, 256, 6, self.Mw, self.Mb)


    def forward(self, x, *y, trainable=False):
        self.samplenumber = x.shape[0]
        #  ============================
        x = self.conv1(x) # 1000 6 24 24
        x = self.SeparableManifoldLayer(x, trainable=trainable)
        x = self.conv2(x)# 1000 2 12 12
        x = self.SeparableManifoldLayer2(x, trainable=trainable)
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
