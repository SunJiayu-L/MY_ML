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

X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

w_norm, b_norm, hist = run_gradient_descent(X_norm, y_train, 1000, 1.0e-1, )

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
print("yp",yp)
print("y_train",y_train)
plt.show()

# First, normalize out example.
x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - X_mu) / X_sigma
print(x_house_norm)
x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")