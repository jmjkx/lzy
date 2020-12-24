import torch
import numpy as np
from math import sqrt

class ManifoldConvLayer(torch.nn.Module):

    def __init__(self, ReductionMethod):
        super().__init__()
        self.ReductionMethod = ReductionMethod

    def forward(self, x, *arg, train=False):
        self.samplenumber = x.shape[0]

        _x = x.cpu().clone().detach().numpy()

        _x = _x.reshape(_x.shape[0] * _x.shape[1], _x.shape[2] * _x.shape[3])

        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2] * x.shape[3])
        if train:
            self.M = self.ReductionMethod(_x, *arg)

            self.M = np.array(self.M)
            self.M = torch.from_numpy(self.M)

            self.weight = torch.nn.Linear(self.M.shape[0], self.M.shape[1], bias=False).double()
            self.weight = self.weight.to('cuda')
            #             self.weight = torch.nn.Parameter(self.M.to('cuda'))

            #             self.weight.requires_grad = True

            #             self.weight.retain_grad()
            self.weight.data = self.M

        x = self.weight(x)
        x = x.reshape(self.samplenumber, int(x.shape[0] / self.samplenumber), int(sqrt(x.shape[1])),
                      int(sqrt(x.shape[1])))

        return x

#     def backward(self, opVec):
#         return torch.mul(self.oneVec, opVec)


# class ManifoldConvLayer(object):                       #  object 和上面的区别   python3内部继承object
#     def __init__(self, ReductionMethod):
#         self.ReductionMethod = ReductionMethod


#     def __call__(self, x, *arg, train = False):
#         _x = x.cpu().clone().detach().numpy()

#         _x = _x.reshape(_x.shape[0]*_x.shape[1], _x.shape[2]*_x.shape[3])
#         if train:

#             self.M = self.ReductionMethod(_x, *arg)
#             self.M = np.array(self.M)

#         __x = np.dot(_x, self.M)
#         __x  = __x.reshape(x.shape[0], int(__x.shape[0] / x.shape[0]), int(sqrt(__x.shape[1])) , int(sqrt(__x.shape[1])))
#         print(self.M)
#         return torch.from_numpy(__x).cuda()
#         x = torch.tensor(x)
#         x = torch.matmul(x.reshape(6000, 576), m1.T)