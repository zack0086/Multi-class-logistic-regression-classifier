'''
Created on 2015-8-28
Edited on 2016-4-16
@author: ZACK
'''

# -*- coding:utf-8 -*-

from __future__ import division
import numpy as np
import time

dataset=[]

def softmax(x):
    e = np.exp(x)
    print e
    if e.ndim == 1:
        return e/np.sum(e,axis=0)
    else:
        sum1 = np.array([np.sum(e, axis=1)]).T
        return e/sum1
    
def train(al,x,y,w_in,b_in):
    l2_regular = 1
    pred_y = softmax(np.dot(x,w_in)+b_in)
    error = y - pred_y
    w_out = w_in + al*np.dot(x.T,error) - al*l2_regular*w_in
    b_out = b_in + al*np.mean(error,axis=0)
    return w_out,b_out

def predict(x,w,b):
    return softmax(np.dot(x,w)+b)

def label2matrix(y_in,k,name):
    m = len(y_in)
    y_out = np.zeros((m,k))
    for j in range(m):
        for i in range(k):
            if y_in[j]==name[i]:
                y_out[j][i] = 1
    return y_out

def choosePre(x):
    index = x.argmax(axis=0)
    x[index] = 1

def findlargest(x):
    index = x.argmax(axis=0)
    return index



#softmax regression function,n means the sum of features,k means the sum of classes
def softmaxReg():
    frTrain = open('optdigits.txt')
    trainingSet = []
    trainingLabels = []
    n = len(frTrain.readline().strip().split('\t')) - 1
    frTrain = open('optdigits.txt')
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr1 = []
        for i in range(n):
            lineArr1.append(float(currLine[i]))
        trainingSet.append(lineArr1)
        trainingLabels.append(currLine[n])
    labelName = []
    for lb in trainingLabels:
        if lb not in labelName:
            labelName.append(lb)
    k = len(labelName)
    m = len(trainingLabels)
    trainingSetMat = np.array(trainingSet)
    trainingLabelMat = label2matrix(trainingLabels,k,labelName)
    alpha = 0.000001
    weights = np.ones((n,k))
    bias = np.zeros(k)
    for i in range(2):
        weights,bias = train(alpha,trainingSetMat,trainingLabelMat,weights,bias)
        alpha = alpha*0.9
    

    y = predict(trainingSetMat,weights,bias)
    for i in xrange(m):
        choosePre(y[i])
    err_cnt = 0
    for i in xrange(m):
        index1 = findlargest(trainingLabelMat[i])
        if y[i][index1] != 1:
            err_cnt += 1
    error_ratio = err_cnt/m
    #print y
    print error_ratio


def run():
    if __name__=="__main__":
        start = time.clock()
        softmaxReg()
        elapsed = (time.clock() - start)
        print "Time used:",elapsed


run()