import copy
import torch
import os
import torch.utils.data as Data
import time
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch import Tensor
import matplotlib as plt
from model import C3DNet

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        # 根据索引获取样本
        threed_input = self.data[index][0]
        output = self.data[index][1]
        return (threed_input, output)

    def __len__(self):
        # 返回数据集大小
        return len(self.data)

def dataloader(batch_size):
    # 数据装载
    training_data = torch.load(r'..\files\train_data.pt')
    training_data = MyDataset(training_data)
    train_data_loader = Data.DataLoader(dataset=training_data,
                                        batch_size=batch_size,
                                        shuffle=True
                                        )
    validation_data = torch.load(r'..\files\val_data.pt')
    validation_data = MyDataset(validation_data)
    val_data_loader = Data.DataLoader(dataset=validation_data,
                                      batch_size=batch_size,
                                      shuffle=True
                                      )
    return train_data_loader, val_data_loader



def gaussian_elimination(A, b):
    n = len(b)

    # 增广矩阵合并
    Ab = np.column_stack((A.astype(float), b.astype(float)))

    # 前向消元
    for i in range(n - 1):
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]

    # 零行消除
    AB = []
    lenth = 0
    for row in Ab:
        if not np.all(row < 0.000001):
            AB.append(row)
            lenth += 1
    AB_array = np.array(AB)

    # 回代求解
    x = np.zeros(n)
    for i in range(lenth-1, -1, -1):
        x[i] = (AB_array[i, n-1] - np.dot(AB_array[i, i:n-1], x[i:n-1])) / AB_array[i, i]
    return x[0:lenth]



# RMSE
def rmse(y_true, y_pred):
    squared_diff = (y_true - y_pred) ** 2
    mean_squared_diff = np.mean(squared_diff)
    rmse_value = np.sqrt(mean_squared_diff)
    return rmse_value

def train_model_process(model, train_data_load, val_data_load, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    #优化器Adam
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    #损失函数使用交叉熵函数
    criterion = nn.MSELoss()

    model = model.to(device)
    #最好（当前）模型权重
    best_model_wts = copy.deepcopy(model.state_dict())

    #参数初始化
    best_acc = float('inf')
    #损失列表
    train_loss_all = []
    val_loss_all = []
    #精度列表
    train_acc_all = []
    val_acc_all = []
    #记录时间
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 30)

        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0.0
        val_corrects = 0
        val_num = 0

        #训练步
        for step, (threed_input, data_output) in enumerate(train_data_load):

            model.train()

            out_put = model(threed_input)#模型输出
            out_label = nn.Softmax(data_output)



            #计算loss
            loss = criterion(out_put, out_label)#loss计算


            #loss反传
            #梯度初始化
            optimizer.zero_grad()
            #梯度计算
            loss.backward()
            #参数更新
            optimizer.step()

            train_loss += loss.items() * data_output.size(0)

            train_corrects += loss

            train_num += data_output.size(0)

        #验证步
        for step, (threed_input, data_output) in enumerate(val_data_load):

            model.eval()

            out_put = model(threed_input)#模型输出
            out_label = nn.Softmax(data_output)
            #计算loss
            loss = criterion(out_put, out_label)#loss计算

            val_loss += loss.items() * data_output.size(0)

            val_corrects += loss

            val_num += data_output.size(0)

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects / val_num)


        print('{} train_loss:{:.4f} train_acc:{:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} val_loss:{:.4f} val_acc:{:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        #准确度提高，更新最优准确度和最优权重
        if val_acc_all[-1] < best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        #计算时间
        used_time = time.time() - since

        print('当前最佳精度:{:.4f}'.format(best_acc))
        print('训练耗时：{:.0f}m{:.0f}s'.format(used_time//60, used_time%60))


        #加载新参数
        model.load_state_dict(best_model_wts)
        #保存权重文件
        torch.save(model.load_state_dict(best_model_wts),r'weight\best_wts.pth')

    train_process = pd.DataFrame(data={'epoch': range(num_epochs),
                                       'train_loss_all':train_loss_all,
                                       'train_acc_all':train_acc_all,
                                       'val_loss_all':val_loss_all,
                                       'val_acc_all':val_acc_all
                                       })

    return train_process


def matplot_loss_acc(train_process):
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, 'ro-', label='train loss')
    plt.plot(train_process['epoch'], train_process.val_loss_all, 'bs-', label='val loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, 'ro-', label='train acc')
    plt.plot(train_process['epoch'], train_process.val_acc_all, 'bs-', label='val acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    batch_size = 10
    C3DNet = C3DNet(batch_size)
    train_data_loader, val_data_loader = dataloader(batch_size)
    train_process = train_model_process(C3DNet, train_data_loader, val_data_loader, 10)
    matplot_loss_acc(train_process)