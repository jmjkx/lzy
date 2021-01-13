import os
import time

import numpy as np
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"


Mw = np.load('/home/liyuan/Programming/python/lzy/graph/graphs/Mw.npy')
Mb = np.load('/home/liyuan/Programming/python/lzy/graph/graphs/Mb.npy')

# deepFeaTrn = deepFeaTrn.cpu().clone().detach().numpy()
deepFea = np.random.rand(60000,512).astype(np.float32)    

deepFeaTrn = torch.from_numpy(deepFea).to('cuda:0')
deepFeaTrn1 = torch.from_numpy(deepFea).to('cuda:1')
Mw = torch.from_numpy(Mw).to('cuda:0')
Mb = torch.from_numpy(Mb).to('cuda:1')

time_start=time.time()
print('==mul==')
A = torch.mm(torch.mm(deepFeaTrn1.T, Mb), deepFeaTrn1)
print('==mul==')
B = torch.mm(torch.mm(deepFeaTrn.T, Mw), deepFeaTrn)

time_end=time.time()

print('totally cost',time_end-time_start)
      
       
