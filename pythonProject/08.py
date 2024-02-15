import random
import os
import torch
from d2l import torch as d2l
import numpy as np
import matplotlib.pyplot as plt

## with torch.no_grad() 则主要是用于停止autograd模块的工作，
# 以起到加速和节省显存的作用，具体行为就是停止gradient计算，从而节省了GPU算力和显存，但是并不会影响dropout和batchnorm层的行为。

# mm只能进行矩阵乘法,也就是输入的两个tensor维度只能是( n × m ) (n\times m)(n×m)和( m × p ) (m\times p)(m×p)
# bmm是两个三维张量相乘, 两个输入tensor维度是( b × n × m )和( b × m × p ), 第一维b代表batch size，输出为( b × n × p )
# matmul可以进行张量乘法, 输入可以是高维.

# python知识补充：
# Python3 range() 函数返回的是一个可迭代对象（类型是对象），而不是列表类型， 所以打印的时候不会打印列表。
# Python3 list() 函数是对象迭代器，可以把range()返回的可迭代对象转为一个列表，返回的变量类型为列表。
# Python3 range(start, stop[, step])
# Python3 shuffle() 方法将序列的所有元素随机排序。shuffle()是不能直接访问的，需要导入 random 模块。举例：random.shuffle (list)
# Python3 yield是python中的生成器


# 人造数据集，creat_data函数中利用正太分布创造张量X（1000行，2列）（服从值为0，标准差为1）,再用生成的X计算出y，y经过处理后转化为列向量
def create_data(w, b, nums_example):
    X = torch.normal(0, 1, (nums_example, len(w)))#这段代码的作用是生成一个包含 num_examples 行、len(w) 列的张量 X，
                                                # 其中的元素是服从均值为 0，标准差为 1 的正态分布的随机数。
    y = torch.matmul(X, w) + b
    print("y_shape:", y.shape)
    y += torch.normal(0, 0.01, y.shape)  # 加入噪声
    return X, y.reshape(-1, 1)  # y从行向量转为列向量，这行代码返回生成的输入特征 X 和目标值 y

#将数值带入，并把数据集赋给features，结果赋给labels
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = create_data(true_w, true_b, 1000)#这意味着 features 变量将包含生成的输入特征数据，
                                                                    # 而 labels 变量将包含对应的目标值数据。

## 读数据集，读取数据集features的个数,然后将0-999随机打乱，每次取一批数据出来，通过yield返回一对features和labels，因为有yield所以每次生成的都被保存起来了。
def read_data(batch_size, features, lables):
    nums_example = len(features)
    indices = list(range(nums_example))  # 生成0-999的元组，然后将range()返回的可迭代对象转为一个列表,range返回的是一个可迭代对象，所以要转化为列表
    random.shuffle(indices)  # 将序列的所有元素随机排序。
    for i in range(0, nums_example, batch_size):  # range(start, stop, step)
        index_tensor = torch.tensor(indices[i: min(i + batch_size, nums_example)])
        yield features[index_tensor], lables[index_tensor]  # 通过索引访问向量

#如果X y都在生成的生成器中，则打印出来
batch_size = 10
for X, y in read_data(batch_size, features, labels):
    print("X:", X, "\ny", y)
    break

##初始化参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)  #这行代码创建了一个形状为 (2, 1) 的张量 w，其中的元素是从均值为 0、
                                                # 标准差为 0.01 的正态分布中随机采样得到的。这个张量被设置为需要梯度计算，因为参数 requires_grad 被设置为 True
b = torch.zeros(1, requires_grad=True)


# 定义模型
def net(X, w, b):
    y=torch.matmul(X, w) + b
    return y


# 定义损失函数
def loss(y_hat, y):
    # print("y_hat_shape:",y_hat.shape,"\ny_shape:",y.shape)
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2  # 这里为什么要加 y_hat_shape: torch.Size([10, 1])  y_shape: torch.Size([10])


# 定义优化  算法 params :给定的所有参数  lr：学习率  这段代码实现了一个批量随机梯度下降的过程，用于更新模型的参数。
def sgd(params, batch_size, lr):
    with torch.no_grad():  # with torch.no_grad() 则主要是用于停止autograd模块的工作，
        for param in params:
            param -= lr * param.grad / batch_size  ##  这里用param = param - lr * param.grad / batch_size会导致导数丢失， zero_()函数报错（更改参数）
            param.grad.zero_()  ## 导数如果丢失了，会报错‘NoneType’ object has no attribute ‘zero_’


# 训练模型  features&labels 已经被初始化完成了（35行），对于某一批X，y，用loss函数计算误差，然后算梯度（ f.sum().backward()），然后优化参数
lr = 0.03
num_epochs = 10

for epoch in range(0, num_epochs):
    for X, y in read_data(batch_size, features, labels):
        f = loss(net(X, w, b), y)
        # 因为`f`形状是(`batch_size`, 1)，而不是一个标量。`f`中的所有元素被加到一起，
        # 并以此计算关于[`w`, `b`]的梯度
        f.sum().backward()
        sgd([w, b], batch_size, lr)  # 使用参数的梯度更新参数
        ##评价模型好坏
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch{epoch+1}, loss { float(train_l.mean())}')
print("w误差 ", true_w - w, "\nb误差 ", true_b - b)