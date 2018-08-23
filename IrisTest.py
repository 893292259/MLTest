import numpy as np  # 导入模块
from sklearn import datasets  # 用数据库去学习，或者把数据库放到tenserflow模块练习
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # 会选择邻近几个点作为他的邻居，综合临近几个点模拟出数据的预测值
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

iris = datasets.load_iris()  # 创建iris的数据，把属性存在X，类别标签存在y
#print(iris)
iris_X = iris.data
iris_y = iris.target


print(iris_X[:4, :])  # print出来iris的属性  每个sample四个属性，描述花的花瓣的长直径等
# print(iris_y)     # 有三个类的花0，1，2

#X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)
# 把所有的data分成了要用来学习的data和用来测试的data   X_test和y_test测试的比例占了总数据的30%

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, random_state=4)

# print(y_train) # 打乱了数据，尽可能的把数据打乱在学习过程中比不乱的更好

knn = KNeighborsClassifier(n_neighbors=5)  # 定义用sklearn中的KNN分类算法

knn.fit(X_train, y_train)  # 用KNN进行数据集的学习，把创建的data放进去，他就自动帮你完成train的步骤
print(knn.predict(X_test))  # 这里的knn就是已经train好了的knn
# 用我的model的属性去预测它是哪一种花
print(y_test)  # 对比真实值
print(knn.score(X_test, y_test))
