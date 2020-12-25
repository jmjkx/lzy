import numpy as np
import scipy.sparse.linalg as sla
from numpy import linalg as LA


def ProjectionMatrix(deepFeaTrn, Mw, Mb, dim):
    deepFeaTrn = deepFeaTrn.cpu().clone().detach().numpy()
    print('======================matmul==============================')
    A = np.matmul(np.matmul(deepFeaTrn.T, Mb), deepFeaTrn)
    B = np.matmul(np.matmul(deepFeaTrn.T, Mw), deepFeaTrn)
    print('======================shape==============================')
    a = A.shape[0]
    b = B.shape[0]
    print('======================eye==============================')
    A = A+np.eye(a)*0.01;
    B = B+np.eye(b)*0.01;
    print('======================TTTTTTTTTTT==============================')
    A=(A+A.T)/2;
    B=(B+B.T)/2;
    # np.save('A.npy', A)
    # np.save('B.npy', B)
    print('eig==============================')
    # A = np.load('A.npy')
    # B = np.load('B.npy')
    eigvalue, eigvector = sla.eigs(A, dim, B, which='LR')
    # for i in range(eigvector.shape[1]):
    #     eigvector[:,i] = eigvector[:,i]./torch.norm(eigvector(:,i));
    #
    # end
    return np.real(eigvector)

def ProjectionMatrix1(deepFeaTrn, Mw, Mb, dim):
    # deepFeaTrn = deepFeaTrn.cpu().clone().detach().numpy()
    # print('======================matmul==============================')
    # A = np.matmul(np.matmul(deepFeaTrn.T, Mb), deepFeaTrn)
    # B = np.matmul(np.matmul(deepFeaTrn.T, Mw), deepFeaTrn)
    # print('======================shape==============================')
    # a = A.shape[0]
    # b = B.shape[0]
    # print('======================eye==============================')
    # A = A+np.eye(a)*0.01;
    # B = B+np.eye(b)*0.01;
    # print('======================TTTTTTTTTTT==============================')
    # A=(A+A.T)/2;
    # B=(B+B.T)/2;
    # np.save('A2.npy', A)
    # np.save('A2.npy', B)
    # print('eig==============================')
    A = np.load('A2.npy')
    B = np.load('B2.npy')
    eigvalue, eigvector = sla.eigs(A, dim, B, which='LR')

    return np.real(eigvector)
