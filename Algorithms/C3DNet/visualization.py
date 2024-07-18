import numpy as np
import pandas as pd

#3D散点图
df = pd.read_csv(r'D:\study\美林实习\datasets\cleandata\Commercial-450-bed Hospital.csv')
series = df[df.columns[0]]
data = series.tolist()
square = []
for i in range(0,len(data),168):
    square.append(data[i:i+168])
matrix_data = np.array(square[0:52])
# print(matrix_data.shape)

#画图
value = []
_index = []
_column = []
for index, i in enumerate(matrix_data):
    for column, j in enumerate(i):
        value.append(j)
        _index.append(index)
        _column.append(column)
print(value,_index,_column)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(_index,_column,value,s=4,c=value)


# 添加坐标轴
plt.show()


