import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

data_train_0 = pd.read_csv('D:/桌面/数据2/Industrial-1-shift Fabricated_Metals.csv', engine='python', skipfooter=3)
data_train = data_train_0[:1000]
d_tr = data_train.astype('float16')

model = auto_arima(d_tr['power [kw]'], start_p=0, max_p=3, d=0, start_q=0, max_q=3,
                                       seasonal=True, m=24,
                                       start_P=3, max_P=4, D=1, start_Q=0, max_Q=2,
                                       trace=True,
                                       error_action='ignore', suppress_warnings=True,
                                       stepwise=True, information_criterion='aic')

data_test_0 =  pd.read_csv('D:/桌面/数据2/Industrial-2-shift Fabricated_Metals.csv', engine='python', skipfooter=3)
data_test = data_test_0[:1000]
d_te = data_test.astype('float16')

# 划分训练集和测试集
input_size = int(len(d_te) * 0.8)
d_input = d_te[:input_size]
d_output = d_te[input_size:]
#train, test = d_t, df[train_size:]

model.fit(d_input['power [kw]'])

# 进行预测
forecast = model.predict( n_periods=len(d_output) )
forecast = pd.DataFrame(forecast, index=d_output.index, columns=['Prediction'])

# 计算均方误差
#mse = np.mean( (d_output['power [kw]'].to_numpy() - forecast['Prediction'].to_numpy()) ** 2 )
mse = mean_squared_error(d_output['power [kw]'].to_numpy(), forecast['Prediction'].to_numpy())
rmse = np.sqrt(mse)

print('RMSE is {:.2f}'.format(rmse))
# 绘制结果
plt.figure(figsize=(15, 7))
plt.plot(d_input.index, d_input['power [kw]'], label='Train')
plt.plot(d_output.index, d_output['power [kw]'], label='Test')
plt.plot(forecast.index, forecast['Prediction'], label='Prediction')
plt.legend()
plt.show()
