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

X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
Y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1,1)  # 2-D Matrix

pos = Y_train == 1
neg = Y_train == 0

pos = Y_train == 1
neg = Y_train == 0

fig,ax = plt.subplots(1,1,figsize=(4,3))
ax.scatter(X_train[pos], Y_train[pos], marker='x', s=80, c = 'red', label="y=1")
ax.scatter(X_train[neg], Y_train[neg], marker='o', s=100, label="y=0", facecolors='none',
              edgecolors=dlc["dlblue"],lw=3)

ax.set_ylim(-0.08,1.1)
ax.set_ylabel('y', fontsize=12)
ax.set_xlabel('x', fontsize=12)
ax.set_title('one variable plot')
ax.legend(fontsize=12)
plt.show()

#这串代码将创建一个包含逻辑层的 Tensorflow 模型，以演示创建模型的替代方法。
# Tensorflow 最常用于创建多层模型。顺序模型是构建这些模型的便捷方法。
model = Sequential(
    [
        tf.keras.layers.Dense(1, input_dim=1,  activation = 'sigmoid', name='L1')
    ]
)

#model.summary() 显示模型中的层数和参数数。
# 此模型中只有一个图层，并且该图层只有一个单元。该单位有两个参数 w b
model.summary()

"""
这段代码是在使用TensorFlow和Keras构建的神经网络模型中，获取特定层的权重和偏置，并打印它们及其形状的过程。

1. `logistic_layer = model.get_layer('L1')`：这行代码通过模型的`get_layer`方法获取了名为`'L1'`的层。
在之前的代码中，我们定义了一个Sequential模型，并且在其中添加了一个名为`'L1'`的Dense层，这个层就是我们在这里获取的对象。

2. `w,b = logistic_layer.get_weights()`：这行代码调用了`logistic_layer`层的`get_weights`方法，该方法返回了一个包含层的所有可训练权重的列表。
对于一个全连接层（Dense层），这个列表通常包含两个元素：第一个是权重矩阵（`w`），第二个是偏置向量（`b`）。在这个例子中，我们使用解包赋值将这两个元素分别赋给了变量`w`和`b`。

3. `print(w,b)`：这行代码打印了获取到的权重矩阵和偏置向量。权重矩阵`w`是一个二维数组，其中包含了输入节点和输出节点之间的权重。偏置向量`b`是一个一维数组，每个元素对应一个输出节点的偏置。

4. `print(w.shape,b.shape)`：这行代码打印了权重矩阵和偏置向量的形状（shape）。`w.shape`给出了权重矩阵的维度，即输入特征的数量和输出单元的数量。
由于我们的模型是一个单层模型，输入特征维度为1，输出单元也为1，所以`w.shape`应该是`(1, 1)`。`b.shape`给出了偏置向量的长度，它等于输出单元的数量。在这个例子中，偏置向量的长度为1，所以`b.shape`应该是`(1,)`。

这段代码的输出将显示模型中Dense层的权重和偏置的具体数值，以及它们的形状信息，这对于理解模型的结构和行为是非常有用的。
"""
logistic_layer = model.get_layer('L1')
w,b = logistic_layer.get_weights()
print(w,b)
print(w.shape,b.shape)

set_w = np.array([[2]])
set_b = np.array([-4.5])
# set_weights takes a list of numpy arrays
logistic_layer.set_weights([set_w, set_b])
print(logistic_layer.get_weights())

a1 = model.predict(X_train[0].reshape(1,1))
print(a1)
alog = sigmoidnp(np.dot(set_w,X_train[0].reshape(1,1)) + set_b)
print(alog)

plt_logistic(X_train, Y_train, model, set_w, set_b, pos, neg)