# 逻辑回归

# 引入数据
from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target
print("类别种类:",np.unique(y))  #打印分类类别的种类

# 切分训练数据和测试数据
## 30%测试数据，70%训练数据，stratify=y表示训练数据和测试数据具有相同的类别比例
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)
# 数据标准化是什么意思？？
sc = StandardScaler()
## 估算训练数据中的mu和sigma
sc.fit(X_train)
## 使用训练数据中的mu和sigma对数据进行标准化
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)



import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap


def plot_decision_region(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolors='black')

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=100.0,random_state=1)
lr.fit(X_train_std,y_train)
print("Class:",lr.classes_)
print("Coef:",lr.coef_)
print("intercept",lr.intercept_)
print("n_iter",lr.n_iter_)


plot_decision_region(X_train_std,y_train,classifier=lr,resolution=0.02)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()