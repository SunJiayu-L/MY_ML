import numpy as np
import matplotlib.pyplot as plt
from plt_one_addpt_onclick import plt_one_addpt_onclick
from lab_utils_common import draw_vthresh
plt.style.use('./deeplearning.mplstyle')

#前置知识：NumPy 有一个名为 exp(x) 的函数，它提供了一种计算输入数组 ( e的下x次方的简便方法）
# Input is an array.
input_array = np.array([1,2,3])
exp_array = np.exp(input_array)

print("Input to exp:", input_array)
print("Output of exp:", exp_array)

# Input is a single number
input_val = 1
exp_val = np.exp(input_val)

print("Input to exp:", input_val)
print("Output of exp:", exp_val)


#sigmoid 函数在 python 中实现
def sigmoid(z):
    """
    Compute the sigmoid of z
    Args:
        z (ndarray): A scalar, numpy array of any size.
    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
    """
    g = 1 / (1 + np.exp(-z))

    return g

# Generate an array of evenly spaced values between -10 and 10
z_tmp = np.arange(-10,11)

# Use the function implemented above to get the sigmoid values
y = sigmoid(z_tmp)

# Code for pretty printing the two arrays next to each other
np.set_printoptions(precision=3)
print("Input (z), Output (sigmoid(z))")
print(np.c_[z_tmp, y])


# Plot z vs sigmoid(z)如你所见，当 z 变为较大负值时，sigmoid 函数接近 0 ；当 z 变为较大正值时，sigmoid 函数接近 1 价值观。
fig,ax = plt.subplots(1,1,figsize=(5,3))#matplotlib中用于创建一个新的图形和坐标轴的函数。
                                                    #1,1 表示创建一个包含1行1列的子图网格，并且只使用第一个子图（即子图索引为0）。
                                                    #figsize=(5,3) 设置了图形的尺寸，这里的5和3分别代表图形的宽度和高度，单位是英寸。
ax.plot(z_tmp, y, c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax,0)
plt.show()

#逻辑回归模型将 sigmoid 应用于熟悉的线性回归模型
#让我们将逻辑回归应用于肿瘤分类的分类数据示例。
#首先，加载示例和参数的初始值。
x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

w_in = np.zeros((1))
b_in = 0

plt.close('all')
addpt = plt_one_addpt_onclick( x_train,y_train, w_in, b_in, logistic=True)
plt.show()