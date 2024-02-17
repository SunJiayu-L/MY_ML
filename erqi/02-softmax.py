import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载Iris数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 对目标标签进行独热编码
n_classes = len(np.unique(y))
y_train_one_hot = np.zeros((len(y_train), n_classes))
y_train_one_hot[np.arange(len(y_train)), y_train] = 1
y_test_one_hot = np.zeros((len(y_test), n_classes))
y_test_one_hot[np.arange(len(y_test)), y_test] = 1

# 初始化模型参数
n_features = X_train.shape[1]
W = np.random.randn(n_features, n_classes)
b = np.zeros((1, n_classes))

# 定义Softmax函数
def softmax(X, W, b):
    Z = np.dot(X, W) + b
    exp_scores = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probabilities

# 定义损失函数（交叉熵损失）
def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-10), axis=1))

# 定义梯度下降函数
def gradient_descent(X, y, W, b, learning_rate, num_iterations):
    for i in range(num_iterations):
        y_pred = softmax(X, W, b)
        loss = cross_entropy_loss(y, y_pred)
        gradients = np.dot(X.T, (y_pred - y)) / len(X)
        dW = gradients
        db = np.sum(gradients, axis=0, keepdims=True) / len(X)
        W -= learning_rate * dW
        b -= learning_rate * db
        print(f"Iteration {i}: Loss = {loss}")

# 训练模型
learning_rate = 0.01
num_iterations = 1000
gradient_descent(X_train, y_train_one_hot, W, b, learning_rate, num_iterations)

# 在测试集上评估模型性能
y_pred_test = softmax(X_test, W, b)
predicted_classes = np.argmax(y_pred_test, axis=1)
test_acc = np.mean(predicted_classes == np.argmax(y_test_one_hot, axis=1))
print(f"Test Accuracy = {test_acc}")