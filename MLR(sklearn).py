'''
Created on 2016-4-18

@author:ZACK
'''

# -*- coding:utf-8 -*-

from __future__ import division
import numpy as np
from sklearn import *

def data_preprocess(data,n):
    dataSet = []
    Labels = []
    m = 0
    for line in data.readlines():
        m +=1
        currLine = line.strip().split('\t')
        lineArr1 = []
        for i in range(n):
            lineArr1.append(float(currLine[i]))
        dataSet.append(lineArr1)
        Labels.append(currLine[n])
    dataSetMat = np.array(dataSet)
    LabelMat = np.array(Labels)
    return dataSetMat,LabelMat,m


def test1(cases):
    data_all = open('testset_3.txt')
    n = len(data_all.readline().strip().split('\t')) - 1
    data_all = open('testset_3.txt')
    x,y,m = data_preprocess(data_all,n)

    case1 = 0.1 #10% for training
    case2 = 0.2 #20% for training
    case3 = 0.3 #30% for training
    case4 = 0.4 #40% for training
    case5 = 0.5 #50% for training
    cs = m * cases

    score_sum = 0
    for i in range(0,10):
        np.random.seed(i)
        indices = np.random.permutation(len(x))

        x_train = x[indices[-cs:]]
        y_train = y[indices[-cs:]]

        x_test = x[indices[:-cs]]
        y_test = y[indices[:-cs]]

        #training
        ifCV = 1
        if(ifCV):
            softmax_model = linear_model.LogisticRegressionCV(solver='newton-cg',multi_class='multinomial',tol=0.0001,cv=10)
        else:
            softmax_model = linear_model.LogisticRegression(solver='newton-cg',multi_class='multinomial',tol=0.0001)
        softmax_model.fit(x_train,y_train)
        score_sum = score_sum + softmax_model.score(x_test,y_test)
    print score_sum/10



def test2():
    data_all = open('dataset_3.txt')
    n1 = len(data_all.readline().strip().split('\t')) - 1
    data_all = open('dataset_3.txt')
    x_train,y_train,m1 = data_preprocess(data_all,n1)

    test_all = open('testset_3.txt')
    n2 = len(test_all.readline().strip().split('\t')) - 1
    test_all = open('testset_3.txt')
    x_test,y_test,m2 = data_preprocess(test_all,n2)

    score_sum = 0
    for i in range(0,10):
        softmax_model = linear_model.LogisticRegression(solver='newton-cg',multi_class='multinomial',tol=0.01)
        softmax_model.fit(x_train,y_train)
        score_sum = score_sum + softmax_model.score(x_test,y_test)
    print score_sum/10

#for cases in range(1,6,1):
    #test1(cases/10)

test2()