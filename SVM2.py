"""本例根据正态分布随机生成两堆点，并求出了之间的区分边际最大化的直线"""
import numpy as np
import pylab as pl
from sklearn import svm

# 设置seed可保证每次出现的随机数不变
np.random.seed(0)
# 20行2列数据，为（2,2）的正态分布，randn函数
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20  # ？？？

clf = svm.SVC(kernel="linear")
clf.fit(X, Y)

# 即为得到w（两个值）的方法
w = clf.coef_[0]
a = -w[0]/w[1]
xx = np.linspace(-5, 5)
# intercept_[0]即为w[3]
yy = a * xx - (clf.intercept_[0]/w[1])

# 这个地方通过方法调出支持向量来b[0], b[1]是第一个点的横纵坐标？？？？？
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
# -1意为取最后一个
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none')
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()
