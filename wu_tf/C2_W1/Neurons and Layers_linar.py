import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import sigmoid
from lab_utils_common import dlc
from lab_neurons_utils import plt_prob_1d, sigmoidnp, plt_linear, plt_logistic
plt.style.use('deeplearning.mplstyle')
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

X_train = np.array([[1.0], [2.0]], dtype=np.float32)           #(size in 1000 square feet)
Y_train = np.array([[300.0], [500.0]], dtype=np.float32)       #(price in 1000s of dollars)

fig, ax = plt.subplots(1,1)
ax.scatter(X_train, Y_train, marker='x', c='r', label="Data Points")
ax.legend( fontsize='xx-large')
ax.set_ylabel('Price (in 1000s of dollars)', fontsize='xx-large')
ax.set_xlabel('Size (1000 sqft)', fontsize='xx-large')
plt.show()

# 线性回归模型
"""在 TensorFlow 中，tf.keras.layers.Dense 是用来创建全连接层（也称为密集层）的类。
您提供的代码片段创建了一个具有单个输出单元（units=1）的全连接层，并且指定了激活函数为 'linear'。
这意味着该层的输出是输入的加权和，没有应用任何非线性激活函数。
"""
linear_layer = tf.keras.layers.Dense(units=1, activation = 'linear', )
#将X_train[0]输入到神经元中，神经元经过线性变换，把输出值赋给a1，此时由于没有赋值wb，所以a1的值并不准确
a1 = linear_layer(X_train[0].reshape(1,1))
print("a1", a1)

w, b= linear_layer.get_weights()
print(f"w = {w}, b={b}")

#配置神经元的w与b
set_w = np.array([[200]])
set_b = np.array([100])

# set_weights takes a list of numpy arrays
linear_layer.set_weights([set_w, set_b])
print(linear_layer.get_weights())

#发现配置好的神经元输出结果与线性回归的一致
a1 = linear_layer(X_train[0].reshape(1,1))
print(a1)
alin = np.dot(set_w,X_train[0].reshape(1,1)) + set_b
print(alin)


#使用神经元完成预测
prediction_tf = linear_layer(X_train)
prediction_np = np.dot( X_train, set_w) + set_b
plt_linear(X_train, Y_train, prediction_tf, prediction_np)

