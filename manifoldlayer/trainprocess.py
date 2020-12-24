import os
# from datapreprocess2 import *
#from introduce import *
#from manifoldConvLayer import ManifoldConvLayer
#from manifoldFcLayer import ManifoldFcLayer
#from myMFA import *
#from myPCA import *
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
# from ipdb import set_trace
from torch import nn

from datainput import extract_data, extract_labels
from datapreprocess import MyDataset
from network import Net

os.environ['CUDA_VISIBLE_DEVICES'] = '8'

import warnings

warnings.filterwarnings('ignore')
############################
import scipy.io as scio
import scipy.io as sio
from tensorboardX import SummaryWriter
from torch.autograd import Variable


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

# Mw = scio.loadmat('graphs/Mw.mat')
#
# Mb = scio.loadmat('graphs/Mb.mat')


#
# Mw = Mw['Mw']
# Mb = Mb['Mb']




# GCN_mask_TR = sample_mask(np.arange(0,695), ALL_Y.shape[0])
# GCN_mask_TE = sample_mask(np.arange(696,10366), ALL_Y.shape[0])
#
# ALL_Y = convert_to_one_hot(ALL_Y - 1, 16)
# ALL_Y = ALL_Y.T
x_train = extract_data(r'/home/liyuan/Programming/python/lzy/graph/data/train-images-idx3-ubyte.gz', 60000).reshape(60000, 1, 28, 28)
y_train = extract_labels(r'/home/liyuan/Programming/python/lzy/graph/data/train-labels-idx1-ubyte.gz', 60000)
x_test = extract_data(r'/home/liyuan/Programming/python/lzy/graph/data/t10k-images-idx3-ubyte.gz', 10000).reshape(10000, 1, 28, 28)
y_test = extract_labels(r'/home/liyuan/Programming/python/lzy/graph/data/t10k-labels-idx1-ubyte.gz', 10000)

#np.save('mnist_trainX.npy')


BATCH_SIZE = 60000
EPOCH = 1000
training_dataset  = MyDataset(x_train, y_train)
testing_dataset = MyDataset(x_test, y_test)
# Data Loaders
train_loader = torch.utils.data.DataLoader(
    dataset=training_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=testing_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# 检查cuda是否可用

# 生成log
now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
log_path = os.path.join(os.getcwd(), "log")
log_dir = os.path.join(log_path, time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

# ---------------------搭建网络--------------------------
cnn = Net()  # 创建CNN, 输入全连接层维数

cnn = cnn.double()

# --------------------设置损失函数和优化器----------------------
optimizer = optim.Adam(cnn.parameters(), lr=0.05)  # lr:(de fault: 1e-3)优化器
criterion = nn.CrossEntropyLoss()  # 损失函数
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=EPOCH / 2, gamma=0.5)  # 设置学习率下降策略

# --------------------训练------------------------------
# 使用GPU
cnn = cnn.cuda()
print('=========STAGE1==============')
for epoch in range(1):
    loss_sigma = 0.0  # 记录一个epoch的loss之和
    correct = 0.0
    total = 0.0
    scheduler.step()  # 更新学习率

    for batch_idx, data in enumerate(train_loader):
        # 获取图片和标签
        inputs, labels = data
        inputs = inputs.double()
        labels = labels.double()
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()  # 清空梯度
        cnn = cnn.train()
        #label_numpy = labels.cpu().numpy()

        outputs = cnn(inputs, train=True)

        #loss = criterion(outputs, torch.max(labels, 1)[1])
        ##         print(cnn.conv1.weight.grad)
        ##         print(cnn.conv1.weight[0][0])
        #loss.backward()  # 反向传播

        #optimizer.step()  # 更新权值s

        #loss_sigma += loss.item()

        ## 统计预测信息
print('=========STAGE2==============')
BATCH_SIZE = 1000
train_loader = torch.utils.data.DataLoader(
    dataset=training_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=testing_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
for epoch in range(EPOCH):
    loss_sigma = 0.0  # 记录一个epoch的loss之和
    correct = 0.0
    total = 0.0
    scheduler.step()  # 更新学习率

    for batch_idx, data in enumerate(train_loader):
        # 获取图片和标签
        inputs, labels = data
        inputs = inputs.double()
        labels = labels.double()
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()  # 清空梯度
        cnn = cnn.train()
        label_numpy = labels.cpu().numpy()

        outputs = cnn(inputs, train=False)

        loss = criterion(outputs, torch.max(labels, 1)[1])
        #         print(cnn.conv1.weight.grad)
        #         print(cnn.conv1.weight[0][0])
        loss.backward()  # 反向传播

        optimizer.step()  # 更新权值s

        loss_sigma += loss.item()

        
        #         print("batch_idx: %s" %batch_idx)
        final_outputs = cnn(torch.from_numpy(x_train).double().cuda(), train=False)  # x_train已经按batch训练过了？？？？
        final_outputs.detach_()
        _, predicted = torch.max(final_outputs, 1)

        total = y_train.shape[0]

        correct = ((predicted == torch.max(torch.from_numpy(y_train).cuda(), 1)[1]).squeeze().sum()).item()

        # 每 BATCH_SIZE 个 iteration 打印一次训练信息，loss为 BATCH_SIZE 个 iteration 的平均
        loss_avg = loss_sigma / BATCH_SIZE

        print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
            epoch + 1, EPOCH, batch_idx + 1, len(train_loader), loss_sigma, correct / total))


    # 记录训练loss
    writer.add_scalars(
        'Loss_group', {'train_loss': loss_avg}, epoch)
    # 记录learning rate
    writer.add_scalar(
        'learning rate', scheduler.get_lr()[0], epoch)
    # 记录Accuracy
    writer.add_scalars('Accuracy_group', {
        'train_acc': correct / total}, epoch)
    # 每个epoch，记录梯度，权值
    #     for name, layer in cnn.named_parameters():
    #         writer.add_histogram(
    #             name + '_grad', layer.grad.cpu().data.numpy(), epoch)
    #         writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

    # ------------------------------------ 观察模型在验证集上的表现 ------------------------------------
    if epoch % 1 == 0:
        loss_sigma = 0.0
        cls_num = len(classes_name)
        conf_mat = np.zeros([cls_num, cls_num])  # 混淆矩阵
        cnn.eval()
        for batch_idx, data in enumerate(test_loader):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            images = images.double()
            labels = labels.double()
            cnn.cuda()
            cnn = cnn.train()

            outputs = cnn(images)  # forward

            outputs.detach_()  # 不求梯度

            loss = criterion(outputs, torch.max(labels, 1)[1])  # 计算loss
            # x = Variable(loss,requires_grad=True)
            # loss = loss.requires_grad()
            loss_sigma += loss.item()

            _, predicted = torch.max(outputs.data, 1)  # 统计
            # labels = labels.data    # Variable --> tensor
            # 统计混淆矩阵
            for j in range(len(labels)):
                cate_i = torch.max(labels, 1)[1][j]
                pre_i = predicted[j]
                conf_mat[int(cate_i), int(pre_i)] += 1.0

        print('{} set Accuracy:{:.2%}'.format(
            'Valid', conf_mat.trace() / conf_mat.sum()))
        # 记录Loss, accuracy
        writer.add_scalars(
            'Loss_group', {'valid_loss': loss_sigma / len(test_loader)}, epoch)
        writer.add_scalars('Accuracy_group', {
            'valid_acc': conf_mat.trace() / conf_mat.sum()}, epoch)
print('Finished Training')

# ----------------------- 保存模型 并且绘制混淆矩阵图 -------------------------
cnn_save_path = os.path.join(log_dir, 'net_params.pkl')
torch.save(cnn.state_dict(), cnn_save_path)

conf_mat_train, train_acc = F(cnn, train_loader, 'train', classes_name)
conf_mat_valid, valid_acc = validate(cnn, test_loader, 'test', classes_name)

show_confMat(conf_mat_train, classes_name, 'train', log_dir)
show_confMat(conf_mat_valid, classes_name, 'valid', log_dir)
