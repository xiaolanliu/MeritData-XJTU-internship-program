import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

daily_gears_charge_actual_L = []
with open('D:/桌面/数据4/全年每日三段收费.csv', 'r') as file: # 365天的三档电价
    reader = csv.reader(file)
    for row in reader:
        daily_gears_charge_actual_L.append([float(row[0]), float(row[1]), float(row[2])])

 # print(daily_gears_charge_actual_L) # 低、中、高三档

hourly_typical_usage_types = [
    [150, 175, 150, 175, 150, 175, 150, 175,
     150, 175, 150, 175, 150, 175, 150, 175,
     150, 175, 150, 175, 150, 175, 150, 175]
    ,
    [150, 175, 150, 175, 150, 175, 150, 175,
     400, 500, 600, 650, 450, 500, 600, 650,
     200, 175, 150, 175, 150, 175, 150, 175]
]
 # 典型天类型，需客户给出，或由客户的时序用电量以聚类生成

typical_day_type_L = [1]*365
for i in range(365):
    if i%7 == 2 or i%7 == 3:
        typical_day_type_L[i] = 0
 # 典型天类型排列，需客户给出

month_days_L = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
month_types_L = [1, 2, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0]
hourly_gears_types = [
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 1, 1, 2],
    [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 1, 1, 1],
]
 # 电价时段，由客户用电地区决定

basic = 0.6
price_types = [basic * 0.5, basic * 1, basic * 1.5]
 # 电价单价，由客户用电地区决定

daily_gears_types_L = []
for i in range(12):
    daily_gears_types_L = daily_gears_types_L + [ month_types_L[i] ] * month_days_L[i] # 全年每日电价单价时序

reconstructed_usage = [] # 下面计算重构的时序用电量
for i in range(365):

    daily_gears_type = daily_gears_types_L[i]  # 当日收费类型
    hourly_gears_L = hourly_gears_types[daily_gears_type]  # 一天24小时的每小时电价档位
    hourly_price_L = [price_types[hour] for hour in hourly_gears_L]  # 一天24小时的每小时电价单价

    typical_day_type = typical_day_type_L[i] # 当日典型天类型
    hourly_typical_usage_L = hourly_typical_usage_types[typical_day_type] # 当日典型天对应的时序用电量

    hourly_price_single_L = [a * b for a, b in zip(hourly_price_L, hourly_typical_usage_L)]
     # 当日典型天对应的时序电费

    three_gears_price_single = [0, 0, 0]  # 当日三档收费
    for j in range(24):
        gear_1 = hourly_gears_L[j]  # 当前小时的档位
        three_gears_price_single[gear_1] = three_gears_price_single[gear_1] + hourly_price_single_L[j]
         # 当日典型天对应的三档电费

    three_gears_price_actual = daily_gears_charge_actual_L[i]
     # 当日真实的三档电费
    three_gears_price_k = [actual / single for actual, single in
                           zip(three_gears_price_actual, three_gears_price_single)]
     # 当日，每一档真实电费与典型天电费之比
    hourly_usage_actual = []
    for k in range(24):
        hourly_usage_actual.append(hourly_typical_usage_L[k] * three_gears_price_k[hourly_gears_L[k]])
     # 当日，每一档真实电费与典型天电费之比，即为每一档真实用电量与典型天用电量之比（假设）
    reconstructed_usage = reconstructed_usage + hourly_usage_actual

ActualData = pd.read_csv('D:/桌面/数据3/Industrial-1-shift Fabricated_Metals1111.csv', encoding='gbk')
actual_L = ActualData['usage'].to_numpy()

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(actual_L, reconstructed_usage)
rmse = np.sqrt(mse) # 计算重构与真实的误差

plt.figure(figsize=(14, 8))
plt.plot(actual_L[:168], color='blue', linewidth=5, label='actual')
plt.plot(reconstructed_usage[:168], color='orange', linewidth=1.5, label='reconstructed')
plt.legend()
plt.title('RMSE is {:.2f}'.format(rmse))
plt.show()
