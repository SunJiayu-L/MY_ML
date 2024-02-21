import os
import pandas as pd
import torch

os.makedirs(os.path.join('..', 'data'), exist_ok=True)  # 创建上级目录中的data文件夹
datafile = os.path.join('..', 'data', 'house_tiny.csv')  # 创建文件
with open(datafile, 'w') as f:  # 往文件中写数据
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 第1行的值
    f.write('2,NA,106000\n')  # 第2行的值
    f.write('4,NA,178100\n')  # 第3行的值
    f.write('NA,NA,140000\n')  # 第4行的值

data = pd.read_csv(datafile)  # 可以看到原始表格中的空值NA被识别成了NaN
print('1.原始数据:\n', data)

inputs, outputs = data.iloc[:, 0: 2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean(numeric_only=True))  # 用均值填充NaN
print("利用求平均值，补充缺失值")
print(inputs)
print("\n")
print(outputs)
print("\n")

# 使用 get_dummies 函数处理输入数据
inputs = pd.get_dummies(inputs, dummy_na=True, dtype=int)

print('2.利用pandas中的get_dummies函数处理:\n', inputs)

# 转换为张量
x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)

print('3.转换为张量：')
print(x)
print(y)
