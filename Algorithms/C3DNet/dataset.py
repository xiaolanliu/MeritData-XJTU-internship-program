import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        # 根据索引获取样本
        return self.data[index]

    def __len__(self):
        # 返回数据集大小
        return len(self.data)

def reshape(data_input):
    daily_group = []
    threed_input = [[[]for _ in range(7)]for _ in range(52)]
    for i in range(0,len(data_input),24):
        daily_group.append(data_input[i:i+24])
    for index, day in enumerate(daily_group):
        threed_input[index // 7][index % 7] = day.tolist()
    threed_input = torch.tensor(threed_input)
    threed_input = threed_input.unsqueeze(0).unsqueeze(0)
    return threed_input

input_windows = 8760
output_windows = 2184
num_samples = 1000
dataset = []
df = pd.read_csv(r'data\2020-2023_hourly.csv')
# print(df)
data = df[df.columns[0]]
# print(data)
Data = pd.concat([data,data[0:input_windows + output_windows]],axis=0)
choice = sorted(random.sample(range(0, len(data), 1), num_samples))
for index, start_index in enumerate(choice):
    print(index)

    groups1 = torch.tensor(np.array(Data[(start_index + 24):(start_index + input_windows)]))
    groups2 = torch.tensor(np.array(Data[(start_index + input_windows):(start_index + input_windows + output_windows)]))
    groups3 = reshape(groups1)

    mean_in, std_in = torch.mean(groups1), torch.std(groups1)
    normalize_in = transforms.Normalize(mean=mean_in, std=std_in)
    threed_input_normalize = normalize_in(groups3)

    groups = [threed_input_normalize, groups2]
    print(threed_input_normalize.size(), groups2.size())
    dataset.append(groups)
# Datasets = torch.tensor(np.array(dataset))
# print(Datasets)
train_data = []
val_data = []
choices = sorted(random.sample(range(0, len(dataset), 1), int(0.2 * len(dataset))))
for index, choice in enumerate(dataset):
    print(index)
    if index in choices:
        val_data.append(choice)
    elif not index in choices:
        train_data.append(choice)
# print(train_data.shape,val_data.shape)
# train_data = torch.tensor(train_data)
# val_data = torch.tensor(val_data)
# print(train_data, val_data)
torch.save(train_data, r'files\train_data.pt')
torch.save(val_data, r'files\val_data.pt')



