#goal :探索平方误差损失不适合逻辑回归的原因
#      :探索逻辑损失函数

import numpy as np
import matplotlib.pyplot as plt
from plt_logistic_loss import  plt_logistic_cost, plt_two_logistic_loss_curves, plt_simple_example
from plt_logistic_loss import soup_bowl, plt_logistic_squared_error
plt.style.use('./deeplearning.mplstyle')

#让我们使用平方误差成本来获得成本的曲面图：可以看到 不同于线性回归的成本函数是一个碗，由于数据离散程度不同，得到的成本曲面图与之前的大不相同
#这产生了一个非常有趣的情节，上面的表面并不像线性回归中的“汤碗”那么光滑！
x_train = np.array([0., 1, 2, 3, 4, 5],dtype=np.longdouble)
y_train = np.array([0,  0, 0, 1, 1, 1],dtype=np.longdouble)
plt_simple_example(x_train, y_train)

plt.close('all')
plt_logistic_squared_error(x_train,y_train)
plt.show()

#逻辑回归需要一个更适合其非线性性质的成本函数。衡量模型好坏的函数是Logistic Loss Function
#函数特点：当预测与目标匹配时为零，当预测与目标不同时，其值迅速增加。考虑以下曲线
plt_two_logistic_loss_curves()

#利用Logistic Loss Function生成的J(w,b)与w b关系图，可以看到非常适合梯度下降
plt.close('all')
cst = plt_logistic_cost(x_train,y_train)
