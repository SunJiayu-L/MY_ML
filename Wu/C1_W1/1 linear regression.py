#问题背景：已知 1平方米300美元，2平方米500美元
#希望通过这两个点拟合线性回归模型，以便预测其他房屋的价格，例如，面积为 1200 平方英尺的房屋。

#思路：1导入库，定义训练数据 2，绘制数据点 3任取w与b，计算y_hat,4利用y_hat与x_train绘图（两点绘图）
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('./deeplearning.mplstyle')

x_train=np.array([1.0,2.0])
y_train=np.array([300.0,500.0])

# 显示数据点
#x,y表示输入数据，marker表示标记的样式，c表示颜色
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()

w=200
b=100

#因为y=wx+b,利用设定的w与b计算y_hat(此处为f_wb)
def compute_model_output(x, w, b):
    m = len(x)
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

#将训练集带入
tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()

w = 200
b = 100
x_i = 1.2
cost_1200sqft = w * x_i + b

print(f"${cost_1200sqft:.0f} thousand dollars")