import torch


class ManifoldFcLayer(torch.nn.Module):
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
            self.weight = torch.nn.Linear(self.M.shape[0], self.M.shape[1], bias=False)
            self.weight = self.weight.to('cuda')
            self.weight.data = self.M

        x = self.weight(x)

        x = x.reshape(self.samplenumber, int(x.shape[0] / self.samplenumber), int(sqrt(x.shape[1])),
                      int(sqrt(x.shape[1])))

        #        set_trace()
        return x