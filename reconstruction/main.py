import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import csv
# 读取数据集，假设数据集已存在并命名为dataset.csv
dataset = pd.read_csv(r"C:\Users\jkljkl\Desktop\LSTM\Industrial-1-shift Fabricated_Metals1111.csv", header=None)

# 将数据集每24个数据划分为一个新的数据，并进行归一化处理
new_data = []
scaler = MinMaxScaler()
for i in range(0, len(dataset), 24):
    data_chunk = np.array(dataset[i:i+24].values).flatten()
    data_chunk_normalized = scaler.fit_transform(data_chunk.reshape(-1, 1)).flatten()
    new_data.append(data_chunk_normalized)

# 对新数据进行KMeans聚类
kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(new_data)

# 计算轮廓系数
silhouette_avg = silhouette_score(new_data, clusters)
print("Average Silhouette Score:", silhouette_avg)

# 输出每一类数据对应的1到365的序号
class_indices = {}
for i, cluster in enumerate(clusters):
    if cluster not in class_indices:
        class_indices[cluster] = []
    class_indices[cluster].append(i+1)

for cluster, indices in class_indices.items():
    print("Cluster", cluster, "Data Indices:", indices)

# 将每一类数据的序号导出为新的数据集
for cluster, indices in class_indices.items():
    cluster_data = {
        "Cluster": [cluster+1] * len(indices),
        "Data Indices": indices
    }
    cluster_df = pd.DataFrame(cluster_data)

# print(cluster_df)
# plt.scatter(range(len(new_data)), clusters, c=clusters, cmap='viridis')
# plt.title("KMeans Clustering Results")
# plt.xlabel("Data Group Index")
# plt.ylabel("Cluster")
# plt.show()
def calculate_electricity_bill(usage_data, price_data):
    total_cost = 0

    for i in range(len(usage_data)):
        for price_slot in price_data:
            if price_slot['start_time'] <= i < price_slot['end_time']:
                total_cost += usage_data[i] * price_slot['price']
                break

    return total_cost

def calculate_hourly_electricity_cost(usage_data, price_data):
    hourly_cost_data = []
    for i in range(len(usage_data)):
        day = i // 24  # 计算当前是一年中的第几天
        price_slot = price_data[day % len(price_data)]  # 根据天数取出对应的电价信息
        hourly_cost = usage_data[i] * price_slot['price']  # 假设用电量单位为kWh
        hourly_cost_data.append(hourly_cost)

    return hourly_cost_data

# 创建长度为一年的用电量数据集，假设为随机数据
dataset=np.array(dataset)
usage_data = dataset

# 电价时段数据，假设电价分为两个时段
price_data = [
    {'start_time': 0, 'end_time': 7, 'price': 0.1},
    {'start_time': 8, 'end_time': 15, 'price': 0.2},  # 第一个时段0-2时，价格0.1元/kWh
    {'start_time': 16, 'end_time': 23, 'price': 0.15},  # 第二个时段3-5时，价格0.15元/kWh

]

hourly_cost_data = calculate_hourly_electricity_cost(usage_data, price_data)
# print("Hourly electricity cost data:")
# for hour, cost in enumerate(hourly_cost_data, 1):
#     print(f"Hour {hour}: {cost}元")
# 保存每小时电费数据到新的数据集
with open('hourly_cost_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(hourly_cost_data)

print("Hourly electricity cost data has been saved to hourly_cost_data.csv")


# 用电量数据

def sum_every_8_data(data):
    sum_data = []
    i = 0
    while i < len(data):
        sum_data.append(sum(data[i:i + 8]))
        i += 8

    return sum_data


# 原始数据集
original_data = hourly_cost_data

# 将原始数据集中每8个数据求和得到新的数据集
sum_data = sum_every_8_data(original_data)
newdata = sum_data
new=np.array(newdata)


# 要导出的List数据集
data = newdata#1095个峰谷平电费

# 指定要写入的CSV文件名
csv_file = r'C:\Users\jkljkl\Desktop\LSTM\峰谷平电费.csv'

# 将List数据集写入CSV文件
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print(f"Data has been exported to {csv_file}")
# 原始数据集


def merge_data(data, group_size):
    merged_data = []
    for i in range(0, len(data), group_size):
        group = data[i:i+group_size]
        merged_data.append(group)
    return merged_data

group_size = 3
group_size1 = 8
dianfei1 = merge_data(newdata, group_size)
dianliang2 = merge_data(dataset, group_size1)
# 输出新的数据集

def chonggou(dianfei,dianliang):
    m=[1, 2, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 26, 27, 28, 29, 30, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44, 47, 48, 49, 50, 51, 54, 55, 56, 57, 58, 61, 62, 63, 64, 65, 68, 69, 70, 71, 72, 75, 76, 77, 78, 79, 82, 83, 84, 85, 86, 89, 90, 91, 92, 93, 96, 97, 98, 99, 100, 103, 104, 105, 106, 107, 110, 111, 112, 113, 114, 117, 118, 119, 120, 121, 124, 125, 126, 127, 128, 131, 132, 133, 134, 135, 138, 139, 140, 141, 142, 145, 146, 147, 148, 149, 152, 153, 154, 155, 156, 159, 160, 161, 162, 163, 166, 167, 168, 169, 170, 173, 174, 175, 176, 177, 180, 181, 182, 183, 184, 187, 188, 189, 190, 191, 194, 195, 196, 197, 198, 201, 202, 203, 204, 205, 208, 209, 210, 211, 212, 215, 216, 217, 218, 219, 222, 223, 224, 225, 226, 229, 230, 231, 232, 233, 236, 237, 238, 239, 240, 243, 244, 245, 246, 247, 250, 251, 252, 253, 254, 257, 258, 259, 260, 261, 264, 265, 266, 267, 268, 271, 272, 273, 274, 275, 278, 279, 280, 281, 282, 285, 286, 287, 288, 289, 292, 293, 294, 295, 296, 299, 300, 301, 302, 303, 306, 307, 308, 309, 310, 313, 314, 315, 316, 317, 320, 321, 322, 323, 324, 327, 328, 329, 330, 331, 334, 335, 336, 337, 338, 341, 342, 343, 344, 345, 348, 349, 350, 351, 352, 355, 356, 357, 358, 359, 362, 363, 364, 365]
    n=[3, 4, 10, 11, 17, 18, 24, 25, 31, 32, 38, 39, 45, 46, 52, 53, 59, 60, 66, 67, 73, 74, 80, 81, 87, 88, 94, 95, 101, 102, 108, 109, 115, 116, 122, 123, 129, 130, 136, 137, 143, 144, 150, 151, 157, 158, 164, 165, 171, 172, 178, 179, 185, 186, 192, 193, 199, 200, 206, 207, 213, 214, 220, 221, 227, 228, 234, 235, 241, 242, 248, 249, 255, 256, 262, 263, 269, 270, 276, 277, 283, 284, 290, 291, 297, 298, 304, 305, 311, 312, 318, 319, 325, 326, 332, 333, 339, 340, 346, 347, 353, 354, 360, 361]
    #print("7",dianliang[7])
    #print("1", dianliang[1])
    chonggou = []
    for i in range(0,365):

        if i+1 in m:
            for k in range(0,3):
                if k==0:
                    zhi=126
                    chonggou.append(((dianfei[i][0]) / zhi) * dianliang[0])
                if k==1:
                    zhi=437.4
                    chonggou.append(((dianfei[i][1]) / zhi) * dianliang[1])
                if k==2:
                    zhi=131.5
                    chonggou.append(((dianfei[i][2]) / zhi) * dianliang[2])


        if i+1 in n:
            for k in range(0, 3):
                if k==0:
                    zhi=189
                    chonggou.append(((dianfei[i][0]) / zhi) * dianliang[6])
                if k==1:
                    zhi=180
                    chonggou.append(((dianfei[i][1]) / zhi) * dianliang[7])
                if k==2:
                    zhi=189
                    chonggou.append(((dianfei[i][2]) / zhi) * dianliang[8])

    return chonggou

data1 = chonggou(dianfei1,dianliang2)#1075个峰谷平电费
data=np.array(data1)
data=data.reshape(8760,1)
# 指定要写入的CSV文件名
csv_file = r'C:\Users\jkljkl\Desktop\LSTM\重构.csv'

# 将List数据集写入CSV文件
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print(f"Data has been exported to {csv_file}")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from math import sqrt
rmse = sqrt(mean_squared_error(data,dataset))
print('Test RMSE: %.3f' % rmse)



