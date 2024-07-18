import numpy as np
import torch
import random
import pandas as pd
import os
from torch.fft import rfft, irfft
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from timefeatures import time_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ETT_Dataloader(object):
    def __init__(self, x, x_stamp, y, train_x, train_stamp, train_y, args, mode='train'):
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.num_var = x.shape[-1]
        self.x, self.x_stamp, self.y = self.get_data(x, x_stamp, y)
        if mode == 'train':
            self.train_x, self.train_stamp = None, None
            self.train_y = None
            self.batch_size = args.batch_size
        else:
            self.train_x, self.train_stamp, self.train_y = self.get_data(train_x[:y.shape[0]], train_stamp[:y.shape[0]], train_y[:y.shape[0]])
            self.batch_size = args.batch_size

        self.mode = mode
        self.x_len = self.x.shape[0]
        self.num_batch = self.x.shape[0] // self.batch_size

    def get_data(self, x, x_stamp, y):
        x = torch.FloatTensor(x).permute(2, 0, 1)
        #x = torch.FloatTensor(x)
        x1 = []
        for i in range(x.shape[0]):
            x1.append(x[i])
        x = torch.cat(x1, dim=0)
        x = x.unsqueeze(-1)
        y = torch.FloatTensor(y).permute(2, 0, 1)
        y1 = []
        for i in range(y.shape[0]):
            y1.append(y[i])
        y = torch.cat(y1, dim=0)
        x_stamp = torch.FloatTensor(x_stamp)
        x_stamp = x_stamp.unsqueeze(-2).repeat(1, 1, self.num_var, 1)
        x_stamp = x_stamp.permute(2, 0, 1, 3)
        x_stamp1 = []
        for i in range(self.num_var):
            x_stamp1.append(x_stamp[i])
        x_stamp = torch.cat(x_stamp1, dim=0)

        return x, x_stamp, y

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                batch_x = self.x[
                             self.current_ind * self.batch_size: min((self.current_ind + 1) * self.batch_size,
                                                                     self.x_len)]
                batch_stamp = self.x_stamp[
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
                    batch_stamp = torch.cat([train_bstamp, batch_stamp], dim=0)
                    batch_y = torch.cat([train_by, batch_y], dim=0)

                yield batch_x.to(device), batch_stamp.to(device), \
                      batch_y.to(device)
                self.current_ind += 1

        return _wrapper()


def get_ETT_sample(args, data, data_stamp):
    num_samples, num_nodes = data.shape
    idx = np.arange(0, num_samples - args.seq_len - args.pred_len)
    np.random.shuffle(idx)
    x, x_stamp, y = [], [], []
    for i in idx:
        x_i = data[i: i + args.seq_len]
        x_stamp_i = data_stamp[i: i + args.seq_len + args.pred_len]
        y_i = data[i: i + args.seq_len + args.pred_len]
        x.append(x_i)
        x_stamp.append(x_stamp_i)
        y.append(y_i)
    x = np.array(x)
    x_stamp = np.array(x_stamp)
    y = np.array(y)
    return x, x_stamp, y

def load_elec_dataset(args):
    all_data = {}
    scaler = StandardScaler()
    df_raw = pd.read_csv(f'data/ETT/{args.data_name}.csv')
    df_stamp = df_raw['Time']
    cols_data = df_raw.columns[0]
    df_data = df_raw[cols_data].values
    df_stamp['date'] = pd.to_datetime(df_stamp)
    data_stamp = time_features(df_stamp, timeenc=1, freq='h')
    border1s = [0, 8 * 30 * 24 - args.seq_len, 10 * 30 * 24 + 2 * 30 * 24 - args.seq_len]
    border2s = [8 * 30 * 24, 8 * 30 * 24 + 2 * 30 * 24, 8760]
    train_data = df_data[border1s[0]:border2s[0]]
    scaler.fit(train_data.reshape(-1, 1))
    data = scaler.transform(df_data.reshape(-1, 1))
    train_x, train_stamp, train_label = get_ETT_sample(args, data[border1s[0]:border2s[0]], data_stamp[border1s[0]:border2s[0]])
    val_x, val_stamp, val_label = get_ETT_sample(args, data[border1s[1]:border2s[1]], data_stamp[border1s[1]:border2s[1]])
    test_x, test_stamp, test_label = get_ETT_sample(args, data[border1s[2]:border2s[2]], data_stamp[border1s[2]:border2s[2]])
    all_data['train_loader'] = ETT_Dataloader(train_x, train_stamp, train_label, None, None, None, args, mode='train')
    all_data['val_loader'] = ETT_Dataloader(val_x, val_stamp, val_label, train_x, train_stamp, train_label, args,
                                            mode='val')
    all_data['test_loader'] = ETT_Dataloader(test_x, test_stamp, test_label, train_x, train_stamp, train_label, args,
                                             mode='test')
    return all_data

def load_ETT_dataset(args):
    all_data = {}
    scaler = StandardScaler()
    df_raw = pd.read_csv(f'data/ETT/{args.data_name}.csv')
    df_stamp = df_raw[['date']]
    cols_data = df_raw.columns[1:]
    df_data = df_raw[cols_data].values
    df_stamp['date'] = pd.to_datetime(df_stamp.date)
    data_stamp = time_features(df_stamp, timeenc=1, freq='h')
    if args.data_name in ['ETTh1', 'ETTh2']:
        border1s = [0, 12 * 30 * 24 - args.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - args.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    else:
        border1s = [0, 12 * 30 * 24 * 4 - args.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - args.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    train_data = df_data[border1s[0]:border2s[0]]
    scaler.fit(train_data)
    data = scaler.transform(df_data)
    train_x, train_stamp, train_label = get_ETT_sample(args, data[border1s[0]:border2s[0]], data_stamp[border1s[0]:border2s[0]])
    val_x, val_stamp, val_label = get_ETT_sample(args, data[border1s[1]:border2s[1]], data_stamp[border1s[1]:border2s[1]])
    test_x, test_stamp, test_label = get_ETT_sample(args, data[border1s[2]:border2s[2]], data_stamp[border1s[2]:border2s[2]])
    all_data['train_loader'] = ETT_Dataloader(train_x, train_stamp, train_label, None, None, None, args, mode='train')
    all_data['val_loader'] = ETT_Dataloader(val_x, val_stamp, val_label, train_x, train_stamp, train_label, args,
                                            mode='val')
    all_data['test_loader'] = ETT_Dataloader(test_x, test_stamp, test_label, train_x, train_stamp, train_label, args,
                                             mode='test')
    return all_data




def divide_data(fea, label, cluster, support_rate):
    cluster = torch.tensor(cluster, device=device)
    B, L, E = fea.shape
    support_len = int(B * support_rate)
    support_fea, support_label = fea[:support_len], label[:support_len]
    query_fea, query_label = fea[support_len:], label[support_len:]
    support_cluster, query_cluster = cluster[:support_len], cluster[support_len:]
    return support_fea, support_label, query_fea, query_label, support_cluster, query_cluster


def autocorrelation(input, dim=0):
    """
    Computes the autocorrelation of samples at dimension ``dim``.
    Reference: https://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation
    """
    if (not input.is_cuda) and (not torch.backends.mkl.is_available()):
        raise NotImplementedError(
            "For CPU tensor, this method is only supported " "with MKL installed."
        )

    # https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/autocorrelation.hpp
    N = input.size(dim)
    M = next_fast_len(N)
    M2 = 2 * M

    # transpose dim with -1 for Fourier transform
    input = input.transpose(dim, -1)

    centered_signal = input - input.mean(dim=-1, keepdim=True)

    # Fourier transform
    freqvec = torch.view_as_real(rfft(centered_signal, n=M2))
    # take square of magnitude of freqvec (or freqvec x freqvec*)
    freqvec_gram = freqvec.pow(2).sum(-1)
    # inverse Fourier transform
    autocorr = irfft(freqvec_gram, n=M2)

    # truncate and normalize the result, then transpose back to original shape
    autocorr = autocorr[..., :N]
    autocorr = autocorr / torch.tensor(range(N, 0, -1), dtype=input.dtype, device=input.device)
    autocorr = autocorr / autocorr[..., :1]
    return autocorr.transpose(dim, -1)


_NEXT_FAST_LEN = {}


def next_fast_len(size):
    """
    Returns the next largest number ``n >= size`` whose prime factors are all
    2, 3, or 5. These sizes are efficient for fast fourier transforms.
    Equivalent to :func:`scipy.fftpack.next_fast_len`.
    :param int size: A positive number.
    :returns: A possibly larger number.
    :rtype int:
    """
    try:
        return _NEXT_FAST_LEN[size]
    except KeyError:
        pass

    assert isinstance(size, int) and size > 0
    next_size = size
    while True:
        remaining = next_size
        for n in (2, 3, 5):
            while remaining % n == 0:
                remaining //= n
        if remaining == 1:
            _NEXT_FAST_LEN[size] = next_size
            return next_size
        next_size += 1

def period_sim(x, y):
    x = x.reshape(-1, x.shape[1])
    y = y.reshape(-1, y.shape[1])
    # input size: batch x length
    """ Autocorrelation """
    x_ac = autocorrelation(x, dim=1)[:, 1:]
    y_ac = autocorrelation(y, dim=1)[:, 1:]

    distance = ((x_ac - y_ac) ** 2).mean(dim=1).mean()

    return -distance

def convert_coeff(x, eps=1e-6):
    x_freq = torch.fft.rfft(x, dim=-1)
    amp = torch.sqrt((x_freq.real + eps).pow(2) + (x_freq.imag + eps).pow(2))
    phase = torch.atan2(x_freq.imag, x_freq.real + eps)
    return x_freq, amp, phase


def fft_sim(x, y):
    x = x.reshape(-1, x.shape[1])
    y = y.reshape(-1, y.shape[1])
    # input size: batch x length
    x_freq, x_amp, x_phase = convert_coeff(x)
    y_freq, y_amp, y_phase = convert_coeff(y)
    amp_d = ((x_amp - y_amp) ** 2).mean(dim=1).mean()
    phase_d = ((x_phase - y_phase) ** 2).mean(dim=1).mean()

    return -(0.01 * amp_d + phase_d)






def shape_extract(x, mode="value"):
    x_diff = x[:, 1:] - x[:, :-1]
    if mode == "binary":
        x_diff[x_diff > 0] = 1.0
        x_diff[x_diff <= 0] = 0.0
        x_diff = x_diff.type(torch.LongTensor).to(x.device)
    return x_diff

def trend_sim(x, y):
    # input size: batch x length
    x = x.reshape(-1, x.shape[1])
    y = y.reshape(-1, y.shape[1])
    x_t = shape_extract(x)
    y_t = shape_extract(y)

    """ The First Order Temporal Correlation Coefficient (CORT) """
    denominator = torch.sqrt(torch.pow(x_t, 2).sum(dim=1)) * torch.sqrt(torch.pow(y_t, 2).sum(dim=1))
    numerator = (x_t * y_t).sum(dim=1)
    denominator[denominator == 0] = 1
    cort = (numerator / denominator)

    return cort.mean()


def log_Normal_standard(x, average=True, dim=None):
    log_normal = -0.5 * torch.pow(x, 2)
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)

def log_Normal_diag(x, mean, var, average=True, dim=None):
    log_normal = -0.5 * (torch.log(var) + torch.pow(x - mean, 2) / ((var ** 2) + 1))
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)
