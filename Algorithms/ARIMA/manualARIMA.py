import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

time_series_0 = pd.read_csv('D:/桌面/数据2/Industrial-Plastic_Manufacturer.csv', engine='python', skipfooter=3)
time_series = time_series_0[:2000]
# 划分训练集和测试集
train_size = int(len(time_series) * 0.9)
train, test = time_series[0:train_size], time_series[train_size:]

# 创建一个窗口，其中包含两个子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15))

# 在第一个子图中绘制ACF图
plot_acf(train, lags=100, ax=ax1)
ax1.set_title('ACF')

# 在第二个子图中绘制PACF图
plot_pacf(train, lags=100, ax=ax2)
ax2.set_title('PACF')

# 显示整个窗口
plt.tight_layout()
plt.show()

# 使用SARIMAX模型进行预测
# 参数(p,d,q)和(P,D,Q)s需要根据ACF和PACF图以及AIC等准则来选取
p, d, q = 0, 0, 0
P, D, Q, s = 4, 1, 1, 24

model = SARIMAX(train,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s))

# 拟合模型
result = model.fit(disp=False)

# 预测未来10个时间点的值
forecast = result.get_forecast(steps=len(test))
forecast_mean = forecast.predicted_mean

# 打印预测结果
#print(forecast_mean)
# 绘制训练集、测试集和预测结果
plt.figure(figsize=(12, 6))
plt.plot(train.index, train.values, label='Train')
plt.plot(test.index, test.values, label='Test')
plt.plot(forecast_mean, color='red', label='Forecast')
plt.legend()
plt.title('Train, Test and Forecast')
plt.show()

#mse = np.mean( (test['power [kw]'].to_numpy() - forecast_mean) ** 2 )
mse = mean_squared_error(test['power [kw]'].to_numpy(), forecast_mean)
rmse = np.sqrt(mse)
print('RMSE is {:.2f}'.format(rmse))