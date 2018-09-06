"""本例单纯从库中调用SVM"""
from sklearn import svm


x = [[2, 0], [1, 1], [2, 3]]
y = [0, 0, 1]
clf = svm.SVC(kernel = 'linear')
clf.fit(x, y)

print(clf)

# get support vectors
print(clf.support_vectors_)
# get indices of support vectors
print(clf.support_)
# get number of support vectors for each class
print(clf.n_support_)
# 这个地方参数有问题很奇怪
print(clf.predict([2, 0]))