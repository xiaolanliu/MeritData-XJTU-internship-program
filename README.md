# 电力数据与时序预测
## 一、简介
本代码库包含：多个已开源的园区电力数据集（类型涵盖轻重工业、食品加工等多个门类），时序预测算法（涵盖arima、CNN-LSTM、3D-CNN、Meta Long-Sequence predication）,面向实际电费账单的还原算法。
## 二、数据集介绍
### 2.1 A 12-month Data of Hourly Energy Consumption Levels from a Commercial-type Consumer ###
能耗数据来自商业园区的智能计量设备，额外附带环境温度气象传感器记录。时间范围为2016全年。  
能级（3_levels、5_levels 和 7_levels）是通过使用统计数据或一些函数将数据切割成 n 个档位（3、5、7） 来产生的。  
**Reference: "Devising Hourly Forecasting Solutions Regarding Electricity Consumption in the Case of Commercial Center Type Consumers." 
(published in Energies 10.11 (2017): 1727) with the new added attributes on its energy levels. https://data.mendeley.com/datasets/n85kwcgt7t/1**


### 2.2 Dataset on Hourly Load Profiles for a Set of 24 Facilities from Industrial, Commercial, and Residential End-use Sectors ###
该数据集包含来自不同最终用途部门（包括工业、商业和住宅消费者）的一组24个代表性设施的一年（8760 小时数据）的每小时负荷曲线。该数据集包括从EnergyPlus参考建筑中采用的六个建筑，
以及通过使用EnergyPlus适应美国新泽西州气候区的18个模拟建筑。  
该数据集可用于对单节点和多节点能源系统进行建模，例如纳米电网、微电网或配电网络中的任何集成系统，其中每栋建筑都由其反映其电力消耗行为的负载分布定义。
**Reference:“Dataset on Hourly Load Profiles for a Set of 24 Facilities from Industrial, Commercial,
and Residential End-use Sectors”, Angizeh, Farhad; Ghofrani, Ali; Jafari, Mohsen A. (2020), Mendeley Data, V1. https://data.mendeley.com/datasets/rfnp2d3kjp/1**


## 三、电费账单还原
1、在reconstruction中，首先将8760小时的每小时用电量数据进行数据集划分，每24行（一天）视作一个新数据，并进行kmeans聚类，计算输出轮廓系数作为聚类判据。  
2、绘图展示聚类结果,为推断全年每小时用电量数据，选取特征天用来做拟合。  
3、根据聚类结果，主要有两种分布的用电量数据，工作日与休息日，因此选取第一个工作日与第一个休息日作为典型天。  
4、根据国家电网披露数据设定一峰谷平电价并计算一年时间每天每小时电费，将每个时段电费求和得到模拟的每天峰谷平电费企业数据集。  
5、遍历该数据，若序号属于工作日一类，即序号在输出的工作日聚类中出现，则视为该天峰谷平时段每小时用电量数据为工作日特征天峰谷平时段用电量数据的放缩，将该天峰谷平电费分别与工作日特征天峰谷平电费相比，比值分别乘以特征天峰谷平三个时段的用电量数据得到该天的重构用电量数据，若该天属于休息日也同理，得到重构数据。  
6、对重构数据与真实数据计算并输出rmse重构误差。  
**原始数据与还原后数据见result中。**

## 四、算法介绍
### 4.1 SARIMA ###
季节性ARIMA（SARIMA）模型是ARIMA的扩展，用于处理具有季节性波动的时间序列数据。  
SARIMA通过在ARIMA模型中加入季节性自回归（SAR）、季节性差分（SD）和季节性移动平均（SMA）项来捕捉季节性模式。  
SARIMA为提高预测准确度，引入额外参数P,D,Q,代表季节性自回归、差分和移动平均的阶数，以及s，表示季节性周期的长度。  
通过这些参数，SARIMA能够同时捕捉时间序列的非季节性和季节性特征，提高预测的准确性。
### 4.2 CNN-LSTM ###
1、LSTM参数设置主要为定义时间序列窗口长度，因为我是用的为每年8760小时用电量数据，因此选择窗口为24是具有物理意义且不会是运算时间过长的。  
2、CNN部分设置主要为卷积核尺寸设置，选择卷积核为24*24，激活函数选择relu函数。设置优化器学习率为0.001，迭代次数自行设置。  
3、每次训练打印训练集损失、验证集损失、训练集百分比误差与验证集百分比误差。  
4、最后绘图展示预测数据与测试数据，并计算各种度量预测准确性的参数，常见数据集均有rmse<10^2。
### 4.3 3D-CNN ###
### 4.4 Meta Long-Sequence predication ###
1、首先，对电力数据集通过FFT将复杂序列解耦为季节和趋势成分，并结合top-k的做法认为季节信号的最高频率的三个分量为噪声，直接去除。  
2、然后，设计了一种自适应的元学习任务构建策略，通过聚类匹配的方法将季节和趋势成分划分到不同的任务中。  
3、最后，使用双流摊销网络( ST-DAN )来捕获季节性趋势任务之间的共享信息，并使用支持集来生成任务特定的参数，以便在查询集上进行快速的泛化学习。  
4、结合上述基于序列分解的自适应元学习概率推理框架，可以有效地增强各种基础模型的长序列预测能力。  
**Reference:Zhu, J., Guo, X., Chen, Y., Yang, Y., Li, W., Jin, B., Wu, F., 2024. Adaptive Meta-Learning Probabilistic Inference Framework for Long Sequence Prediction. AAAI 38, 17159–17166. https://doi.org/10.1609/aaai.v38i15.29661**
## 五、总结
