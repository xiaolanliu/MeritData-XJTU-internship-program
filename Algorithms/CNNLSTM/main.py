import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
# 调用GPU加速
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

dataframe = pd.read_csv(r"C:\Users\jkljkl\Desktop\LSTM\8years.csv", usecols=[1], engine='python', skipfooter=3)

# 将整型变为float
dataset = dataframe.astype('float32')
#归一化
data = pd.DataFrame(dataset)

print(data)
print(type(data))
# 选择特征, 共6列特征
feats = data.iloc[:,0:]
# 对离散的星期几的数据进行onehot编码

# 特征列增加到12项
print(feats)  # (348, 12)

# 选择标签数据，一组时间序列预测5天后的真实气温
pre_days = 8760
# 选择特征数据中的真实气温'actual'具体向上移动5天的气温信息
targets = feats['power [kw]'].shift(-pre_days)
# 查看标签信息


# 由于特征值最后5行对应的标签是空值nan，将最后5行特征及标签删除
feats = feats[:-pre_days]
targets = targets[:-pre_days]
# 查看数据信息
print('feats.shape:', feats.shape, 'targets.shape:', targets.shape)  # (343, 12) (343,)


# 特征数据标准化处理
from sklearn.preprocessing import StandardScaler
# 接收标准化方法
scaler = StandardScaler()
# 对特征数据中所有的数值类型的数据进行标准化
feats.iloc[:,:5] = scaler.fit_transform(feats.iloc[:,:5])
# 查看标准化后的信息
print(feats)

import numpy as np
from collections import deque  # 队列，可在两端增删元素

# 将特征数据从df类型转为numpy类型
feats = np.array(feats)

# 定义时间序列窗口是连续10天的特征数据
max_series_days = 24
# 创建一个队列，队列的最大长度固定为10
deq = deque(maxlen=max_series_days)  # 如果长度超出了10，先从队列头部开始删除

# 创建一个列表，保存处理后的特征序列
x = []
# 遍历每一行数据，包含12项特征
for i in feats:
    # 将每一行数据存入队列中, numpy类型转为list类型
    deq.append(list(i))
    # 如果队列长度等于指定的序列长度，就保存这个序列
    # 如果队列长度大于序列长度，队列会自动删除头端元素，在尾端追加新元素
    if len(deq) == max_series_days:
        # 保存每一组时间序列, 队列类型转为list类型
        x.append(list(deq))

# 保存与特征对应的标签值
y = targets[max_series_days-1:].values

# 保证序列长度和标签长度相同

print(y)

# 将list类型转为numpy类型
x, y = np.array(x), np.array(y)
total_num = len(x)  # 一共有多少组序列
train_num = int(total_num*0.8)  # 前80%的数据用来训练
val_num = int(total_num*0.9)  # 前80%-90%的数据用来训练验证
# 剩余数据用来测试

x_train, y_train = x[:train_num], y[:train_num]  # 训练集
x_val, y_val = x[train_num: val_num], y[train_num: val_num]  # 验证集
x_test, y_test = x[val_num:], y[val_num:]  # 测试集

# 构造数据集
batch_size = 10  # 每次迭代处理128个序列
# 训练集
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.batch(batch_size).shuffle(10000)
# 验证集
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_ds = val_ds.batch(batch_size)
# 测试集
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.batch(batch_size)

# 查看数据集信息
sample = next(iter(train_ds))  # 取出一个batch的数据
print('x_train.shape:', sample[0].shape)  # (128, 10, 12)
print('y_train.shape:', sample[1].shape)  # (128,)
# 输入层要和x_train的shape一致，但注意不要batch维度
input_shape = sample[0].shape[1:]  # [10,12]

# 构造输入层
inputs = keras.Input(shape=(input_shape))  # [None, 10, 12]

# 调整维度 [None,10,12]==>[None,10,12,1]
x = layers.Reshape(target_shape=(inputs.shape[1], inputs.shape[2], 1))(inputs)

# 卷积+BN+Relu  [None,10,12,1]==>[None,10,12,8]
x = layers.Conv2D(8, kernel_size=(24,24), strides=1, padding='same', use_bias=False,
                  kernel_regularizer=keras.regularizers.l2(0.01))(x)

x = layers.BatchNormalization()(x)  # 批标准化
x = layers.Activation('relu')(x)  # relu激活函数

# 池化下采样 [None,10,12,8]==>[None,10,6,8]
x = layers.MaxPool2D(pool_size=(1,1))(x)

# 1*1卷积调整通道数 [None,10,6,8]==>[None,10,6,1]
x = layers.Conv2D(1, kernel_size=(24,24), strides=1, padding='same', use_bias=False,
                  kernel_regularizer=keras.regularizers.l2(0.01))(x)

# 把最后一个维度挤压掉 [None,10,6,1]==>[None,10,6]
x = tf.squeeze(x, axis=-1)

# [None,10,6] ==> [None,10,16]
# 第一个LSTM层, 如果下一层还是LSTM层就需要return_sequences=True, 否则就是False
x = layers.LSTM(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
x = layers.Dropout(0.2)(x)  # 随机杀死神经元防止过拟合

# 输出层 [None,16]==>[None,1]
outputs = layers.Dense(1)(x)

# 构建模型
model = keras.Model(inputs, outputs)

# 查看模型架构
model.summary()

# 网络编译
model.compile(optimizer = keras.optimizers.Adam(0.001),  # adam优化器学习率0.001
              loss = tf.keras.losses.MeanAbsoluteError(),  # 标签和预测之间绝对差异的平均值
              metrics = tf.keras.losses.MeanSquaredLogarithmicError())  # 计算标签和预测之间的对数误差均方值。

epochs = 1  # 迭代300次

# 网络训练, history保存训练时的信息
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)
history_dict = history.history  # 获取训练的数据字典
train_loss = history_dict['loss']  # 训练集损失
val_loss = history_dict['val_loss']  # 验证集损失
train_msle = history_dict['mean_squared_logarithmic_error']  # 训练集的百分比误差
val_msle = history_dict['val_mean_squared_logarithmic_error']  # 验证集的百分比误差
import matplotlib.pyplot as plt
# （11）绘制训练损失和验证损失
plt.figure()
plt.plot(range(epochs), train_loss, label='train_loss')  # 训练集损失
plt.plot(range(epochs), val_loss, label='val_loss')  # 验证集损失
plt.legend()  # 显示标签
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

# （12）绘制训练百分比误差和验证百分比误差
plt.figure()
plt.plot(range(epochs), train_msle, label='train_msle')  # 训练集指标
plt.plot(range(epochs), val_msle, label='val_msle')  # 验证集指标
plt.legend()  # 显示标签
plt.xlabel('epochs')
plt.ylabel('msle')
plt.show()

# 对整个测试集评估
model.evaluate(test_ds)

# 预测
y_pred = model.predict(x_test)



# 绘制对比曲线
fig = plt.figure(figsize=(10,5))  # 画板大小
ax = fig.add_subplot(111)  # 画板上添加一张图
# 绘制真实值曲线
ax.plot( y_test)
# 绘制预测值曲线
ax.plot( y_pred[1:])
# 设置x轴刻度
plt.show()

r_2 = r2_score(y_pred,y_test)
print('Test r_2: %.3f' % r_2)
# 计算MAE
mae = mean_absolute_error(y_pred,y_test)
print('Test MAE: %.3f' % mae)
# 计算RMSE
from math import sqrt
rmse = sqrt(mean_squared_error(y_pred,y_test))
print('Test RMSE: %.3f' % rmse)
print(y_pred)
print(y_pred.shape)
print(type(y_pred))