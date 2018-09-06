# from NN1 import NeuralNetwork
# import numpy as np
#
# nn = NeuralNetwork([2, 2, 1],"tanh")
# X = np.array([[0, 0], [0, 1], [1, 0]])
# y = np.array([0, 1, 1])
# nn.fit(X, y)
# for i in [[0, 0], [1, 0], [0, 1], [1, 1]]:
#     print(i, nn.predict(i))
from sklearn.datasets import  load_digits

digits = load_digits()
import  pylab as pl
pl.gray()
pl.matshow(digits.images[1])
pl.show()