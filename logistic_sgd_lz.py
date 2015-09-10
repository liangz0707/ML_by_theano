# -*- coding: utf-8 -*-
__author__ = 'LiangZhe'

'''
    最简单的有监督机器学习内容:多元线性回归。
    通过tx和ty进行训练；
    得到参数矩阵W和b，通过梯度下降法求解，代价函数为cost；
'''
import theano
import theano.tensor as T
import numpy 

print('computing begining!')
n_in ,n_out = [5,3]
tx=[ [18,4,1,2,3],
    [19,4,2,1,0],
    [15,3,2,2,1],
    [12,3,1,2,1],
    [2,3,12,2,1],
    [1,1,34,2,1],
    [2,1,14,1,2],
    [2,3,13,2,1],
    [2,1,1,2,11],
    [1,3,6,2,22],
    [2,2,3,2,8],
    [1,3,1,2,13]
    ]
ty = [
        [18,0,0],
        [18,0,0],
        [13,0,0],
        [12,0,0],
        [0,12,0],
        [0,32,0],
        [0,12,0],
        [0,14,0],
        [0,0,12],
        [0,0,32],
        [0,0,12],
        [0,0,22],
    ]
x = T.vector()  # data, presented as rasterized images
y = T.vector() 

W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
b = theano.shared(
            value=numpy.zeros(
                (n_out, ),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

print(W.get_value())
print(b.get_value())

print()
index = T.lscalar() 
borrow=True

test_x = theano.shared(numpy.asarray(tx,dtype=theano.config.floatX),borrow=borrow)
test_y = theano.shared(numpy.asarray(ty,dtype=theano.config.floatX),borrow=borrow)

cost = T.mean((T.dot(x,W)+b - y)**2)

gW = T.grad(cost,W)
gb = T.grad(cost,b)

learning_rate=0.002

updates = [(W, W - learning_rate * gW),
               (b, b - learning_rate * gb)]

train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: test_x[index],
            y: test_y[index]
        }
    )

result = T.dot(x,W)+b
test_model = theano.function(
        inputs=[x],
        outputs=result
    )

for i in range(1,120):
    #print(W.get_value())
    #print(b.get_value())
    train_model(i%12)
print(W.get_value())
print(b.get_value())
print()
print(test_model([1,3,22,2,3]))
print(test_model([1,3,2,2,23]))
print(test_model([19,3,2,2,3]))
print()
print(test_model([1,3,100,2,3]))
print(test_model([1,3,2,2,9]))
print(test_model([9,3,2,2,3]))
print()
print(test_model([1,23,1,2,3]))
print(test_model([1,3,2,32,9]))