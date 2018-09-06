"""多元线性回归举例"""
from numpy import genfromtxt
import  numpy as np
from sklearn import datasets, linear_model

datapath = r"I:\pycharm\data\truck.csv"
deliverdata = genfromtxt(datapath, delimiter=',')

X = deliverdata[:, :-1]
Y = deliverdata[:, -1]

regr = linear_model.LinearRegression()

regr.fit(X, Y)

print(regr.coef_)