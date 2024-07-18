import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#库

data_0 = pd.read_csv('D:/桌面/数据3/区域负荷2020-2023数据.csv', encoding='gbk')#导入数据
data_0['总有功功率（kw）'] = data_0['总有功功率（kw）']/4  #如果为15min数据，假设每一段功率恒定，用功率计算用电量
data_0.columns = ['Date', 'Usage_kWh']

plt.figure(figsize=(16, 8))
plt.plot(data_0['Usage_kWh'][:168*2])
plt.show()
#查看用电量部分图像

data_1 = data_0[['Date', 'Usage_kWh']][1:]#第一行0:00数据属于上一小时的最后15min，无法构成完整的一小时，因此去掉
data_1 = data_1.reset_index(drop=True)#重置行索引
print('data_1')
print(data_1)

new_row_indices = [ i*4+3 for i in range( int(data_1.shape[0]/4) ) ]
data_2 = data_1.iloc[new_row_indices]
data_2 = data_2.reset_index(drop=True)
for i in range( int(data_1.shape[0]/4) ):
    for j in range(3):
        data_2.iloc[i, 1] = data_2.iloc[i, 1] + data_1.iloc[4*i+j, 1]#将四个15min相加生成新的1h数据
#data_2['Date'] = data_2['Date'].str.replace('00:00', '24:00')
print('data_2')
print(data_2)

#data_2.to_csv('D:/桌面/数据3/2020-2023_hourly.csv', index=False)#导出数据
