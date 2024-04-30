
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt
dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'
plt.style.use('./deeplearning.mplstyle')
from lab_utils_multi import  load_house_data, compute_cost, run_gradient_descent
from lab_utils_multi import  norm_plot, plt_contour_multi, plt_equal_scale, plot_cost_i_w
from sklearn.preprocessing import scale

X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']

# 让我们通过绘制每个特征与价格的关系来查看数据集及其特征。
# 绘制每个功能与目标价格的关系图，可以提供一些关于哪些功能对价格影响最大的指示。
# 如上所述，尺寸的增加也会导致价格的上涨。卧室和地板似乎对价格没有太大影响。新房子的价格比旧房子高。
fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price (1000's)")
plt.show()

# 探讨学习率𝛼对收敛的影响
_, _, hist = run_gradient_descent(X_train, y_train, 10, alpha = 9.9e-7)
plot_cost_i_w(X_train, y_train, hist)

_,_,hist = run_gradient_descent(X_train, y_train, 10, alpha = 9e-7)
plot_cost_i_w(X_train, y_train, hist)

_,_,hist = run_gradient_descent(X_train, y_train, 10, alpha = 1e-7)
plot_cost_i_w(X_train, y_train, hist)

#
def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column

    Args:
      X (ndarray): Shape (m,n) input data, m examples, n features

    Returns:
      X_norm (ndarray): Shape (m,n)  input normalized by column     X_norm 是通过 zscore_normalize_features 函数处理后的数据集
      mu (ndarray):     Shape (n,)   mean of each feature每个特征的均值
      sigma (ndarray):  Shape (n,)   standard deviation of each feature每个特征的标准差
    在这个函数中，X_norm 的计算步骤如下：
    首先，计算输入数据 X 中每个特征（列）的均值 mu。
    然后，计算每个特征的标准差 sigma。
    对于 X 中的每个样本，从其每个特征值中减去该特征的均值，然后除以该特征的标准差。这个操作是逐元素（element-wise）进行的。
    最终得到的 X_norm 是一个形状与 X 相同的数组，但其特征值已经标准化。
    """

    # find the mean of each column/feature
    mu = np.mean(X, axis=0)  # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma = np.std(X, axis=0)  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma

    return (X_norm, mu, sigma)


mu     = np.mean(X_train,axis=0)
sigma  = np.std(X_train,axis=0)
X_mean = (X_train - mu)
X_norm = (X_train - mu)/sigma

""""
左：未标准化：“尺寸（平方英尺）”特征的值范围或方差远大于年龄的范围
中：第一步查找从每个特征中删除平均值。这留下了以零为中心的特征。很难看出“年龄”特征的差异，但“尺寸（平方英尺）”显然在零左右。
右：第二步除以方差。这使得两个特征都以零为中心，具有相似的尺度。
"""
#让我们看一下 Z-score标准化所涉及的步骤。下图显示了逐步的转变。
fig,ax=plt.subplots(1, 3, figsize=(12, 3))
ax[0].scatter(X_train[:,0], X_train[:,3])
ax[0].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3])
ax[0].set_title("unnormalized")
ax[0].axis('equal')

ax[1].scatter(X_mean[:,0], X_mean[:,3])
ax[1].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3])
ax[1].set_title(r"X - $\mu$")
ax[1].axis('equal')

ax[2].scatter(X_norm[:,0], X_norm[:,3])
ax[2].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[2].set_title(r"Z-score normalized")
ax[2].axis('equal')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle("distribution of features before, during, after normalization")
plt.show()

#让我们对数据进行标准化并将其与原始数据进行比较。
#通过归一化，每列的峰峰值范围从数千倍减少到 2-3 倍。
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

#请注意，上面标准化数据的范围以零为中心，大致为+/- 1。最重要的是，每个特征的范围都是相似的。
fig,ax=plt.subplots(1, 4, figsize=(12, 3))
for i in range(len(ax)):
    norm_plot(ax[i],X_train[:,i],)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count");
fig.suptitle("distribution of features before normalization")
plt.show()
fig,ax=plt.subplots(1,4,figsize=(12,3))
for i in range(len(ax)):
    norm_plot(ax[i],X_norm[:,i],)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count");
fig.suptitle(f"distribution of features after normalization")

plt.show()

#缩放后的特征可以更快地获得非常准确的结果！请注意，在这个相当短的运行结束时，每个参数的梯度都很小。 0.1 的学习率是使用归一化特征进行回归的良好开端。
w_norm, b_norm, hist = run_gradient_descent(X_norm, y_train, 1000, 1.0e-1, )

#让我们绘制预测值与目标值的关系图。请注意，预测是使用归一化特征进行的，而绘图是使用原始特征值显示的
#predict target using normalized features
m = X_norm.shape[0]
yp = np.zeros(m)
for i in range(m):
    yp[i] = np.dot(X_norm[i], w_norm) + b_norm

    # plot predictions and targets versus original features
fig,ax=plt.subplots(1,4,figsize=(12, 3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],yp,color=dlorange, label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend()
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()

# First, normalize out example.
x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - X_mu) / X_sigma
print(x_house_norm)
x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")

#在下图中，参数的比例是匹配的。左图是 w[0]（平方英尺）与 w[1]（标准化特征之前的卧室数量）的成本等值线图。
# 该图非常不对称，以至于看不到完成轮廓的曲线。相反，当特征标准化时，成本轮廓更加对称。结果是，在梯度下降期间更新参数可以使每个参数取得相同的进展。
plt_equal_scale(X_train, X_norm, y_train)