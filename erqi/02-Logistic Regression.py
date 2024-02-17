from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 加载Iris数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建多分类逻辑回归模型
# 对于多分类问题，通常指定solver='lbfgs'或'liblinear'，并且需要设置multi_class参数
#solver='lbfgs'：指定了逻辑回归模型的求解器。在这里，lbfgs 是一种拟牛顿方法的优化算法，用于最小化损失函数。它在处理小型数据集时表现良好，通常是默认的选择之一。
#multi_class='multinomial'：指定了逻辑回归模型的多分类策略。在这里，multinomial 表示采用多项式逻辑回归，即使用softmax函数进行多类别分类。
# 对于多分类问题，通常可以选择 multinomial 或 ovr（一对多）策略。
# max_iter=1000：指定了算法运行的最大迭代次数。逻辑回归模型是通过迭代优化算法来拟合数据和更新模型参数的。max_iter 参数控制了算法运行的最大迭代次数，
# 如果在指定迭代次数内算法没有收敛（达到停止条件），则会提前结束。在这里，将最大迭代次数设置为 1000。
logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)

# 训练模型
logreg.fit(X_train, y_train)

# 预测测试集
y_pred = logreg.predict(X_test)

# 输出准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 输出分类报告
print(classification_report(y_test, y_pred))