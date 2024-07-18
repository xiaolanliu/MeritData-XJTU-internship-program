C3DNet for Time Series Data Prediction——用于时序数据预测的C3DNet网络
==========================================
## Background

本项目背景源自某能源服务公司，该客户旨在通过一年长度1小时为单位的电力消耗数据，对客户所在园区能源设备的采购和长期的储能调度提供方案和建议，由于客户往往只能提供过往数据，数据分布与未来存在差异，需要从过往数据中归纳预测未来某一时段的用电情况，进而为决策算法提供良好的数据支持。

## Introduction

本项目旨在对某园区一年长度1小时为单位的电力消耗数据进行处理从一个过去的[8760]序列重建一个未来的[8760]序列，通过分析典型数据集可以发现，该数据总体呈现周期性波动，以168小时（即1周）为周期，且在24小时（1天）周期上也有较强的自相似性。故将数据[8760]进行重构，舍弃其中的[24]个数据，将余下[8736]个数据重构为[52，7，24]结构（即52周、7天、24小时），使用3D版本的CNN网络（即C3DNet）进行分析，最终实现预测。

由于训练成本有限，为了有效控制模型参数量，同时平衡步进预测对长序列预测结果置信度的影响，模型设置为单次预测[2184]长度即1季度13周的数据，通过4次步进的方式实现从[8760]序列重建[8760]序列的过程。

模型结构大致如下：
```chatinput
x = x[0]
transformation = trans(x)
x = self.sig(self.Conv1(x))
x = self.sig(self.Conv2(x))
x = self.Pool3(x)
x = self.flatten(x)
x = self.sig(self.linear4(x))
x = self.sig(self.linear5(x))
x = self.sig(self.linear6(x))
x = torch.einsum('bj,bjk->bk', [x, transformation])
```
其中transformation是由input填充的稀疏转换矩阵，用于将模型输出的权重label转换成[2184]的预测结果。模型的直接输出结果为[13*52]，其中每52个out_label表示从input中52周的数据加权和的权值，得到[168]的1周预测结果，共计13周，通过拼接获得[2184]的结果。理论上可以通过扩张全连接层实现一次预测[8760]全年长度序列。

## Usage
### File Description
[timeseriespredction](../timeseriespredction)<br>
│  [dataclean.py](dataclean.py)  对数据进行预清洗<br>
│  [dataset.py](dataset.py)  构成数据集<br>
│  [README.md](README.md)  Readme<br>
│  [visualization.py](visualization.py)  可视化程序<br>
├─[C3DNet](C3DNet)  C3DNet模型<br>
│  │  [model.py](C3DNet/model.py)  模型<br>
│  └─ [train.py](C3DNet/train.py)  训练程序<br>
├─[data](data)  存放原始数据集<br>
│  └─  [2020-2023_hourly.csv](data/2020-2023_hourly.csv)  原始数据集<br>
├─[files](files)  存放训练和验证数据集<br>
│  │   [train_data.pt](files/train_data.pt)  训练数据<br>
│  └─  [val_data.pt](files/val_data.pt)  验证数据<br>
└─