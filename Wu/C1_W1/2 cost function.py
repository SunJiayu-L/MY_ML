import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl  # 确保这些函数存在于您的项目中

plt.style.use('./deeplearning.mplstyle')

#引入数据 #  a house with 1000 square feet sold for $300,000
#           and a house with 2000 square feet sold for $500,000.
#x_train = np.array([1.0, 2.0])
#y_train = np.array([300.0, 500.0])
x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])

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

plt_intuition(x_train,y_train)

plt.close('all')
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)
soup_bowl()

