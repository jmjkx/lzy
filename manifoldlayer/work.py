import torch
import torch.nn as nn
import torch.optim as optim


class MyDense(nn.Module):
    def __init__(self):
        super(MyDense, self).__init__()
        self.params = nn.ParameterList([])
        self.params.append(nn.Parameter(torch.rand((4 ,4), requires_grad=True)))
        # self.params.append(nn.Parameter(torch.randn(4, 4)))
        # self.params1 = nn.ParameterDict({
        #         'linear1': nn.Parameter(torch.randn(4, 4))
        # })
        # self.params1.update({'linear2': nn.Parameter(torch.randn(4, 2))}) # 新增

    def forward(self, x):
        # x = self.params(x)
        x = x[0, 0, :]
        x = torch.mul(x, self.params[0])
        return x
        


X = torch.rand(10, 3, 3, 4)
Y = torch.rand(10, 4)
criterion = nn.MSELoss()
net = MyDense()
optimizer = optim.Adam(net.parameters(), lr=0.05)  # lr:(de fault: 1e-3)优化器
for i in range(10):
    optimizer.zero_grad()
    output = net(X[i])
    loss = criterion(output, Y[i].unsqueeze(1))
    
    
    loss.backward()  # 反向传播
    optimizer.step()  # 更新权值s
    print(1)
    # print(next(net.params.parameters()))
    print(next(net.params.parameters()))


   # def singlechannelconv(self, x: torch.tensor, *manifoldlayer: torch.nn.modules.linear.Linear)-> torch.tensor:
    #     x = x.reshape(x.shape[0], x.shape[2]*x.shape[2])
    #     if self.trainable:
    #         print('===========WWWWWWWWW=================')
    #         W = self.ProjectionMatrix(x, self.Mw, self.Mb, self.dim).astype(np.float64)
    #         self.manifold = torch.nn.Linear(W.shape[0], W.shape[1], bias=False).double()
    #         self.manifold.data = torch.from_numpy(W.astype(np.float64))
    #         self.manifold = self.manifold.to('cuda')
    #         # self.W = nn.Parameter(torch.from_numpy(W).to('cuda'), requires_grad=True)
    #         self.params.append(self.manifold)
    #     else:
    #         self.manifold = manifoldlayer[0]
    #     # x = torch.mm(x, self.W)
    #     x = self.manifold(x)
    #     x = x.reshape(x.shape[0], int(sqrt(x.shape[1])), \
    #         int(sqrt(x.shape[1])))
    #     return x
