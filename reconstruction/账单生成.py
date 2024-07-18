import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

data_0 = pd.read_csv('D:/桌面/数据3/Industrial-1-shift Fabricated_Metals1111.csv', encoding='gbk')
hourly_usage_L = data_0['usage'].to_numpy()

plt.figure(figsize=(12, 6))
plt.plot(hourly_usage_L[:168], linewidth=0.5)
plt.title('USAGE')
plt.show()

hourly_typical_usage = [
                        [150, 175, 150, 175, 150, 175, 150, 175,
                         150, 175, 150, 175, 150, 175, 150, 175,
                         150, 175, 150, 175, 150, 175, 150, 175]
                        ,
                        [150, 175, 150, 175, 150, 175, 150, 175,
                         400, 500, 600, 650, 450, 500, 600, 650,
                         200, 175, 150, 175, 150, 175, 150, 175]
                       ]
#人为设定典型天

month_days_L = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
month_types_L = [1, 2, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0]
hourly_gear_type = [
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 1, 1, 2],
    [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 1, 1, 1],
]  # 电价时段有三类
hourly_gear_L = []
for i in range(12):
    hourly_gear_L = hourly_gear_L + hourly_gear_type[month_types_L[i]] * month_days_L[i]  # 全年每小时电价档位
plt.figure(figsize=(12, 6))
plt.plot(hourly_gear_L[:168], linewidth=0.5)
plt.title('GEAR')
plt.show()

basic = 0.6
price_type = [basic*0.5, basic*1, basic*1.5]
hourly_price_L = [price_type[hour] for hour in hourly_gear_L] # 全年每小时电价单价
plt.figure(figsize=(12, 6))
plt.plot(hourly_price_L[:168], linewidth=0.5)
plt.title('PRICE')
plt.show()

hourly_charge_L = [usage*price for usage,price in zip(hourly_usage_L,hourly_price_L)]# 全年每小时电价总价
plt.figure(figsize=(12, 6))
plt.plot(hourly_charge_L[:168], linewidth=0.5)
plt.title('CHARGE')
plt.show()

daily_gears_charge_L = []
for i in range(365):
    daily_gears_charge = [0, 0, 0]  # 低、中、高三档
    for j in range(24):
        date = 24 * i + j
        gear = hourly_gear_L[date]
        charge = hourly_charge_L[date]
        daily_gears_charge[gear] = daily_gears_charge[gear] + charge
    daily_gears_charge_L.append(daily_gears_charge)
 # 将每天电价归纳为三档

with open('D:/桌面/数据4/全年每日三段收费.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for row in daily_gears_charge_L:
        writer.writerow(row)
 # 将365天的三档电价保存为csv