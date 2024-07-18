import os
import numpy as np
import pandas as pd
import time

def mktime(lenth):
    Time = []
    # start = time.mktime((2020, 1, 1, 0, 0, 0, 0, 0, 0))
    start = time.mktime((2010, 1, 1, 0, 0, 0, 0, 0, 0))
    for i in range(lenth):
        time_str = start + 3600 * i
        time_sec = time.strftime("%Y %m %d %H:%M:%S",time.localtime(time_str))
        Time.append(time_sec)
    return Time

#文件读取

datasets = 'aliyun'
Storage = 'D:\study\美林实习\datasets'

Folder_path = os.path.join(Storage, datasets)
File_name = os.listdir(Folder_path)
print(File_name)

def detect_outliers(data, threshold=3):#3σ清洗
    mean = np.mean(data)
    std = np.std(data)
    print(mean,std)
    Data = []
    for value in data:
        z_score = (value - mean) / std
        if abs(z_score) > threshold:
            Data.append(mean)
        else:
            Data.append(value)
    return Data

#数据清洗
Clean_data = []
for index, n in enumerate(File_name):
    df = pd.read_csv(Folder_path + "/" + n)
    series = df[df.columns[1]]
    data = series.tolist()
    Data = detect_outliers(data)
    data_series = pd.Series(Data)
    data_series = data_series.interpolate(method='linear')#缺失值填充
    Time = mktime(len(series))
    dataframe = pd.DataFrame(data_series)
    dataframe = dataframe.assign(c=Time)
    dataframe.columns = ['power [kw]','Time']
    # print(str(index) + 'finished')
    Clean_data.append(dataframe) 
    dataframe.to_csv(Storage + r'\cleandata/' + n,index=False)
# print(Clean_data)


#训练集测试集划分
