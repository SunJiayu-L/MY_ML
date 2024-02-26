import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl  # 确保这些函数存在于您的项目中

plt.style.use('./deeplearning.mplstyle')

#引入数据 #  a house with 1000 square feet sold for $300,000
#           and a house with 2000 square feet sold for $500,000.
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

#计算成本函数，遍历每一个数据
def cost_function(x,y,w,b):
    m=len(x)
    cost_sum=0
    for i in range(m):
        f_wb=w*x[i]+b
        cost=(f_wb-y[i])**2
        cost_sum=cost_sum+cost
    total_cost = (1 / (2 * m)) * cost_sum

    return  total_cost

#f_dw 即为y_hat
#计算偏导数，输入值为x训练集，y目标集，wb为假设斜率与截距，返回偏导数dj_dw,dj_db
def computer_gradient(x,y,w,b):
    m=len(x)
    dj_dw=0
    dj_db=0
    for i in range(m):
        f_wb=w*x[i]+b
        dj_dw_i=(f_wb-y[i])*x[i]
        dj_db_i=f_wb-y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return  dj_dw,dj_db

