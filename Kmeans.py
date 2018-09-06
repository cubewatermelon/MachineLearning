"""本例实现了Kmeans算法"""
import numpy as np


def kmeans(X, k, maxIt):
    numPoints, numDim = X.shape
    # 扩展一列作为标记
    dataSet = np.zeros((numPoints, numDim + 1))
    dataSet[:, :-1] = X
    # 随机选择k个中心点
    centroids = dataSet[np.random.randint(numPoints, size=k), :]
    centroids[:, -1] = range(1, k + 1)

    print(centroids)
    # 计数用
    iteration = 0
    oldcentroids = None

    while not shouldStop(oldcentroids, centroids, iteration, maxIt):
        oldcentroids = np.copy(centroids)
        iteration += 1
        # 更新每个数据的所属标签
        updateLabels(dataSet, centroids)
        # 更新每个类的中心点
        centroids = getCentroids(dataSet, k)

    return dataSet


def shouldStop(oldCentroids, centroids, iteration, maxIt):
    if iteration > maxIt:
        return True
    return np.array_equal(oldCentroids, centroids)


def updateLabels(dataSet, centroids):
    numPoints, numDim = dataSet.shape
    # 一行行刷新标签
    for i in range(0, numPoints):
        dataSet[i, -1] = getLabelFromCloestCentroids(dataSet[i, :-1], centroids)


def getLabelFromCloestCentroids(dataSetRow, centroids):
    label = centroids[0, -1]
    minDist = np.linalg.norm(dataSetRow - centroids[0, :-1])
    for i in range(1, centroids.shape[0]):
        dist = np.linalg.norm(dataSetRow - centroids[i, -1])
        if dist < minDist:
            minDist = dist
            label = centroids[i, -1]

    return label


def getCentroids(dataSet, k):
    result = np.zeros((k, dataSet.shape[1]))
    for i in range(1, k+1):
        oneCluster = dataSet[dataSet[:, -1] == i, :-1]
        # 压缩行，对各列求均值
        result[i - 1, :-1] = np.mean(oneCluster, axis=0)
        result[i - 1, -1] = i

    return result


x1 = np.array([1, 1])
x2 = np.array([2, 1])
x3 = np.array([4, 5])
x4 = np.array([5, 6])
testX = np.vstack((x1, x2, x3, x4))

print(testX)
result = kmeans(testX, 2, 10)
print(result)
