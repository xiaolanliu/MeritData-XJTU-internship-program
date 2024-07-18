import pandas as pd

# 读取原始数据
input_file = 'data/ETT/Commercial-450-bed Hospital.csv'  # 输入文件路径
data = pd.read_csv(input_file)

# 将数据重整为365*24的格式
reshaped_data = data.values.reshape(365, 24)

# 将结果保存为新的CSV文件
output_file = 'output_data.csv'  # 输出文件路径
pd.DataFrame(reshaped_data).to_csv(output_file, index=False, header=False)

print("数据转换完成，已保存为新的CSV文件。")
