import csv
import matplotlib.pyplot as plt

def csv_read(CSV_data,CSV_number):#第CSV_data工况,第CSV_number组轴承
    data_csv = []
    data_H = []#水平传感器测量的诊断数据的数组
    data_L = []#垂直传感器测量的诊断数据的数组
    CSV = [[123, 161, 158, 122, 52], [491, 161, 533, 42, 339], [2538, 2496, 371, 1515, 114]]#3种工况分别都有5个轴承，CSV数据集样本总数
    CSV_path = ["", "35Hz12kN", "37.5Hz11kN", "40Hz10kN"]
    #样本数
    # 35Hz12kN   1  1-123   2-161   3-158  4-122   5-52
    # 37.5Hz11kN 2  1-491   2-161   3-533  4-42    5-339
    # 40Hz10kN   3  1-2538  2-2496  3-371  4-1515  5-114
    path = "D://XJTU-SY//Data//XJTU-SY_Bearing_Datasets//" + CSV_path[CSV_data] + "//Bearing" + str(CSV_data) + "_" + str(CSV_number) + "//"
    print(path)
    for i in range(50,CSV[CSV_data-1][CSV_number-1]):#二维数组从0开始，显示部分周期修改range(里的1)
        csv_data=csv.reader(open(path+"%d.csv"% i,"r"))
        for list in csv_data:
            data_csv.append(list)
        for j in range(1, len(data_csv)):
            data_H.append(float(data_csv[j][1]))
            data_L.append(float(data_csv[j][0]))
        data_csv = []
    return data_H, data_L

data1,data2=csv_read(1,5)#第一种工况，第5组轴承的数据
#显示全周期,非全周期则是在for i in range(50,CSV[CSV_data-1][CSV_number-1]):显示部分周期修改range(里的1)
import matplotlib.pyplot as plt
plt.figure()
plt.subplot(121)
plt.title("Horizontal_vibration_signals")
plt.ylabel("Amplitude")
plt.xlabel("t")
plt.plot(data1)
plt.subplot(122)
plt.title("Vertical_vibration_signals")
plt.ylabel("Amplitude")
plt.xlabel("t")
plt.plot(data2)
plt.show()
