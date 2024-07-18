import numpy as np
import torch
import random
import pandas as pd
import os
import torch
import argparse
import numpy as np
import warnings
import os
from utils import load_XJTU_dataset
from models import Model
from torch.fft import rfft, irfft
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from timefeatures import time_features
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class XJTU_Dataloader(object):
    def __init__(self, x, y, train_x,  train_y, args, mode='train'):
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.num_var = x.shape[-1]
        self.x,  self.y = self.get_data(x, y)
        if mode == 'train':
            self.train_x, self.train_stamp = None, None
            self.train_y = None
            self.batch_size = args.batch_size
        else:
            self.train_x, self.train_y = self.get_data(train_x[:y.shape[0]], train_y[:y.shape[0]])
            self.batch_size = args.batch_size

        self.mode = mode
        self.x_len = self.x.shape[0]


        self.num_batch = self.x.shape[0] // self.batch_size

    def get_data(self, x, y):
        #print("train_loader中处理前的x", x.shape)
        x = torch.FloatTensor(x).permute(2, 0, 1)
        x1 = []
        #print("train_loader中处理后的x", x.shape)
        for i in range(x.shape[0]):
            x1.append(x[i])        #重写了一遍x
        x = torch.cat(x1, dim=0)
        x = x.unsqueeze(-1)
        #print("train_loader中拼接后的x", x.shape)
        y = torch.FloatTensor(y).permute(2, 0, 1)
        y1 = []
        for i in range(y.shape[0]):
            y1.append(y[i])
        y = torch.cat(y1, dim=0)
        return x,  y

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                batch_x = self.x[
                             self.current_ind * self.batch_size: min((self.current_ind + 1) * self.batch_size,
                                                                     self.x_len)]

                batch_y = self.y[
                             self.current_ind * self.batch_size: min((self.current_ind + 1) * self.batch_size,
                                                                     self.x_len)]
                if self.mode != 'train':
                    train_bx = self.train_x[
                             self.current_ind * self.batch_size: min((self.current_ind + 1) * self.batch_size,
                                                                     self.x_len)]
                    train_bstamp = self.train_stamp[
                             self.current_ind * self.batch_size: min((self.current_ind + 1) * self.batch_size,
                                                                     self.x_len)]
                    train_by = self.train_y[
                             self.current_ind * self.batch_size: min((self.current_ind + 1) * self.batch_size,
                                                                     self.x_len)]
                    batch_x = torch.cat([train_bx, batch_x], dim=0)
                    batch_y = torch.cat([train_by, batch_y], dim=0)

                yield batch_x.to(device),  \
                      batch_y.to(device)
                self.current_ind += 1

        return _wrapper()
def get_XJTU_sample(args, data):
    num_samples, num_nodes = data.shape
    idx = np.arange(0, num_samples - args.seq_len - args.pred_len)
    np.random.shuffle(idx)
    x, y = [], []
    for i in idx:
        x_i = data[i: i + args.seq_len]
        y_i = data[i: i + args.seq_len + args.pred_len]
        x.append(x_i)
        y.append(y_i)
    x = np.array(x)
    y = np.array(y)
    #print("get_XJTU", x.shape)
    return x,  y


def load_XJTU_dataset(args):
    all_data = {}
    scaler = StandardScaler()
    data_csv = []
    data_H = []#水平传感器测量的诊断数据的数组
    data_L = []#垂直传感器测量的诊断数据的数组
    CSV = [[123, 161, 158, 122, 52], [491, 161, 533, 42, 339], [2538, 2496, 371, 1515, 114]]#3种工况分别都有5个轴承，CSV数据集样本总数
    CSV_path = ["", "35Hz12kN", "37.5Hz11kN", "40Hz10kN"]
    path = "XJTU-SY//Data//XJTU-SY_Bearing_Datasets//" + CSV_path[args.CSV_data] + "//Bearing" + str(
        args.CSV_data) + "_" + str(args.CSV_number) + "//"+"%d.csv"% args.CSV_datanumber
    print(path)
    df_raw = pd.read_csv(path)
    cols_data = df_raw.columns[0:]
    df_data = df_raw[cols_data].values
    if 1:
        border1s = [0, 12 * 30 * 24 - args.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - args.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

    train_data = df_data[border1s[0]:border2s[0]]
    scaler.fit(train_data)
    data = scaler.transform(df_data)
    train_x, train_label = get_XJTU_sample(args, data[border1s[0]:border2s[0]])
    val_x, val_label = get_XJTU_sample(args, data[border1s[1]:border2s[1]])
    test_x, test_label = get_XJTU_sample(args, data[border1s[2]:border2s[2]])
    all_data['train_loader'] = XJTU_Dataloader(train_x, train_label, None, None, args, mode='train')
    all_data['val_loader'] = XJTU_Dataloader(val_x,  val_label, train_x,  train_label, args,
                                            mode='val')
    all_data['test_loader'] = XJTU_Dataloader(test_x,  test_label, train_x,  train_label, args,
                                             mode='test')
    #print("train_loader中的x",all_data['train_loader'].x.shape)
    #print("train_loader中的y",all_data['train_loader'].y.shape)
    return all_data


def divide_data(fea, label, cluster, support_rate):
    cluster = torch.tensor(cluster, device=device)
    B, L, E = fea.shape
    support_len = int(B * support_rate)
    support_fea, support_label = fea[:support_len], label[:support_len]
    query_fea, query_label = fea[support_len:], label[support_len:]
    support_cluster, query_cluster = cluster[:support_len], cluster[support_len:]
    return support_fea, support_label, query_fea, query_label, support_cluster, query_cluster

def Args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='ETTh2', type=str,
                        choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'], help='dataset type')
    parser.add_argument('--base_name', default='AMPIF', type=str, help='')
    parser.add_argument('--batch_size', default=64, type=float, help='batch size of train input data')
    parser.add_argument('--base_lr', default=0.000002, type=float, help='optimizer learning rate')
    parser.add_argument('--epoch', default=100, type=float, help='Number of training epoch')
    parser.add_argument('--show_len', default=300, type=float,
                        help='Output the results every how many batches during the training process')
    parser.add_argument('--epoch_begin', default=0, type=float, help='Starting epoch')

    parser.add_argument('--optimizer', default='adam', type=str, help='Optimizer')
    parser.add_argument('--seed', default=3407, type=float, help='Random seed setting')
    parser.add_argument('--max_grad_norm', default=5, type=float, help='Gradient cropping')
    parser.add_argument('--dropout', default=0, type=float, help='dropout')
    parser.add_argument('--epsilon', default=1.0e-4, type=float, help='optimizer epsilon')
    parser.add_argument('--emb_dim', default=128, type=float, help='Embedding Dimension')
    parser.add_argument('--hid_dim', default=128, type=float, help='Hidden Dimension')
    parser.add_argument('--conv_dim', default=8, type=float, help='Convolutional layer dimension')
    parser.add_argument('--fc_dim', default=696, type=float, help='Global mapping linear layer dimension')
    parser.add_argument('--kernel_size', default=10, type=float, help='kernel_size')
    parser.add_argument('--seq_len', default=96, type=float, help='input sequence length')
    parser.add_argument('--pred_len', default=24, type=float, help='prediction sequence length')

    parser.add_argument('--support_rate', default=0.5, type=float, help='Support rate')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--factor', type=int, default=1, help='')
    parser.add_argument('--num_cluster', default=20, type=float, help='')
    parser.add_argument('--num_samples', default=10, type=float, help='Number of task parameter samples')
    parser.add_argument('--low_f_num', default=3, type=float, help='')
    parser.add_argument('--high_f_num', default=8, type=float, help='')

    parser.add_argument('--is_low_noise', default=True, type=bool, help='')
    parser.add_argument('--shuffle', default=True, type=bool, help='')
    parser.add_argument('--only_test', default=False, type=bool, help='status')


    parser.add_argument('--CSV_data', default=1, type=int)
    parser.add_argument('--CSV_number', default=1, type=int)  #1-1工况
    parser.add_argument('--CSV_datanumber', default=1, type=int)

    args = parser.parse_args()
    args.output_name = "dm_%s_bs_%d_lr_%f_dim_%d_nc_%d_ns_%d" % (args.data_name, args.batch_size, args.base_lr,
                                                                      args.emb_dim, args.num_cluster, args.num_samples)

    return args
args = Args()
data = load_XJTU_dataset(args)
print(data)