import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing    # 获取加州房价数据集
from sklearn.model_selection import train_test_split #train_test_split 函数用于将数据集拆分为训练集和测试集，
from sklearn.linear_model import LinearRegression #用于创建线性回归模型
from sklearn.metrics import mean_squared_error, r2_score #函数用于评估模型的性能。

# 加载波士顿房价数据集
boston = fetch_california_housing()
X = boston.data
y = boston.target
print(y)

# 数据集 X 和 y 拆分成了训练集和测试集，其中训练集占总数据集的 80%，测试集占总数据集的 20%，并且设置了随机种子为 42。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用线性回归模型进行训练
model = LinearRegression()
model.fit(X_train, y_train)

diabetes_y_pred = model.predict(X_test)

# 获取斜率和截距
slope = model.coef_
intercept = model.intercept_

# 打印斜率和截距
print("Slope (weights):", slope)
print("Intercept (bias):", intercept)




# 计算模型的性能指标
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)

# 绘制散点图
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("True Values")
plt.ylabel("Predictions")

# 绘制对角线
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=3)
plt.show()

def score(self,y_pred,y):
    accuracy = (y == y_pred).sum/len(y)
    return accuracy

""""
y_train_pred = model.predict(X_train)
y_train_pred_reshape = y_train_pred.reshape(-1,1)
y_test_pred = model.predict(X_test)
y_test_pred_reshape = y_test_pred.reshape(-1,1)
print(y_test_pred_reshape)
y_train_reshape= y_train.reshape(-1,1)
y_test_reshape= y_test.reshape(-1,1)
score_train=model.score(y_train_reshape, y_train_pred_reshape)
score_test=model.score(y_test_reshape, y_test_pred_reshape)

#print("score_train",score_train)
#print("score_test",score_test)
"""