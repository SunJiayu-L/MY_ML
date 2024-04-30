#这个代码加入了之前的一些东西 图变多了而已  主要还是看4 Multiple Variable Linear Regression
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_gradients, plt_contour_wgrad, plt_divergence  # 确保这些函数存在于您的项目中
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

#左一显示了  ∂𝐽(𝑤,𝑏)∂𝑤成本曲线  𝑤相对于三个点的斜率。在图的右侧，导数为正数，而在左侧为负数。
# 由于“碗形”，导数将始终导致梯度下降到梯度为零的底部。
#右边的图已经固定  𝑏=100了。梯度下降将同时利用两者  ∂𝐽(𝑤,𝑏)∂𝑤，∂𝐽(𝑤,𝑏)∂𝑏来更新参数。
#右侧的“箭袋图”提供了一种查看两个参数梯度的方法。箭头大小反映了该点的梯度大小。箭头的方向和斜率反映了该点的  ∂𝐽(𝑤,𝑏)∂𝑤比率  ∂𝐽(𝑤,𝑏)∂𝑏。
#请注意，梯度点远离最小值。缩放的梯度从𝑤或𝑏的当前值 中减去。这会将参数向降低成本的方向移动。
plt_gradients(x_train,y_train, cost_function, computer_gradient)

plt.show()

"""
Performs gradient descent to fit w,b. Updates w,b by taking 
num_iters gradient steps with learning rate alpha

Args:
  x (ndarray (m,))  : Data, m examples 
  y (ndarray (m,))  : target values
  w_in,b_in (scalar): initial values of model parameters  
  alpha (float):     Learning rate
  num_iters (int):   number of iterations to run gradient descent
  cost_function:     function to call to produce cost
  gradient_function: function to call to produce gradient

Returns:
  w (scalar): Updated value of parameter after running gradient descent
  b (scalar): Updated value of parameter after running gradient descent
  J_history (List): History of cost values
  p_history (list): History of parameters [w,b] 
  """
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    w = copy.deepcopy(w_in)  # avoid modifying global w_in
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        # 获得偏导数
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # 更新参数
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        # Save cost J at each iteration保存每次迭代完J的数据
        if i < 100000:  # prevent resource exhaustion 为了避免资源耗尽（例如内存或计算资源），在实际应用中，梯度下降算法可能会运行很多次迭代，尤其是在处理大型数据集或者复杂的模型时。
                    # 如果每次都记录和打印成本函数的值，可能会产生大量的数据，这不仅会占用大量的内存空间，还可能导致程序运行缓慢。
                    # 因此，通过设置一个合理的迭代次数上限，可以有效地控制资源的使用，防止程序因为资源不足而崩溃。
            J_history.append(cost_function(x, y, w, b))#保存成本函数J
            p_history.append([w, b])#保存每一次的wb
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w, b, J_history, p_history  # return w and J,w history for graphing

#函数实现
# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings（迭代次数与学习率）
iterations = 10000
tmp_alpha = 1.0e-2
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha,
                                                    iterations, cost_function, computer_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

#绘图
# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.plot(x_train, w_final*x_train+b_final, c='b',label='Our Prediction')
plt.show()

# 梯度下降的成本与迭代次数：刚开始时，少量迭代可以使成本函数骤降，结束时，大量迭代带来的成本函数变化非常小

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step')
plt.show()


print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand dollars")

fig1, ax = plt.subplots(1,1, figsize=(12, 6))
plt_contour_wgrad(x_train, y_train, p_hist, ax)
plt.show()

fig2, ax = plt.subplots(1,1, figsize=(12, 4))
plt_contour_wgrad(x_train, y_train, p_hist, ax, w_range=[180, 220, 0.5], b_range=[80, 120, 0.5],
            contours=[1,5,10,20],resolution=0.5)
plt.show()


# initialize parameters
w_init = 0
b_init = 0
# set alpha to a large value
iterations = 10
tmp_alpha = 8.0e-1
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha,
                                                    iterations, cost_function, computer_gradient)
plt_divergence(p_hist, J_hist,x_train, y_train)
plt.show()
