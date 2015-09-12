# -*- coding: utf-8 -*-
__author__ = 'LiangZhe'

''' 
    最简单的多层感知器。
    使用了early stoping和L1、L2防止过拟合。
'''
import cPickle
import gzip

import os
import sys
import timeit

import theano
import theano.tensor as T
import numpy 

#定义隐藏层类，进入隐藏层需要通过s(W*x+b)的计算。
class HiddenLayer(object):
    '''
        rng:用作初始化W参数矩阵
        input:是输入的数据类型
        n_in:输入数据的维数
        n_out:输出数据的维数
        W，b：是否制定初始的学习参数
        activation：激活函数（tanh或sigmoid）
    '''
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):

        self.input = input

        if W is None:
            #如果没有就随机的生成，并且转换成需要的
            W_values = numpy.asarray( 
                    rng.uniform(
                        low= -numpy.sqrt(6.0/n_in+n_out),
                        high= numpy.sqrt(6.0/n_in+n_out),
                        size=(n_in,n_out)
                    ),
                    dtype=theano.config.floatX
                )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            #设置成共享数据
            #W = theano.shared(value=W_values, name='W', borrow=True)
            W = theano.shared(W_values,borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,),dtype=theano.config.floatX)
            b = theano.shared(b_values,borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        #如果有激活函数就是用没有就直接使用结果。
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        #保存参数
        self.params = [self.W, self.b]

#通过逻辑回归层作为输出层:
class LogisticRegression(object):
    def __init__(self,input,n_in,n_out):
        self.W =theano.shared(
                    value=numpy.zeros((n_in,n_out),dtype=theano.config.floatX),
                    borrow=True
                )
        self.b =theano.shared(
                    value=numpy.zeros((n_out,),dtype=theano.config.floatX),
                    borrow=True
                )
        #每个结果的概率
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        #预测结果
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

        self.input = input
    #定义似然函数
    def negative_log_likelihood(self, y):
        #p_y_given_x就是当前各个结果的似然函数。这里提取除了结果对应的几个，求最小（因为取了负数，本身应该去最小）
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])

    #定义误差函数
    def errors(self,y):
        #暂时不做多余检查。误差：就是有几何结果和原来不同。
        return T.mean(T.neq(self.y_pred, y))


#定义多层感知网络对象。
class MLP(object):
    def __init__(self,rng,input,n_in,n_hidden,n_out): 
        #隐藏层
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in =n_in,
            n_out=n_hidden,
            activation=T.tanh
        )
        #输出层
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        #防止过拟合的约束L1和L2
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        self.L2 = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )
        #似然概率直接用输出层的。
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )

        #误差也直接使用输出层的。
        self.errors = self.logRegressionLayer.errors
        #将两个参数组合起来。只是简单的放在一起“+”并不做运算。
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        #输入就是整个网络的输入。
        self.input = input

#加载数据
def load_data(dataset):
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)

    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def shared_dataset(data_xy, borrow=True):

        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)

        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

#网络的使用。
def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
    #加载数据集，并将数据集放在share内存中
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    #计算batches的个数
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    #进行训练：
    index = T.lscalar() 
    x = T.matrix('x') 
    y = T.ivector('y')
    #随机数生成器主要用来生成隐藏层的初始参数。
    rng = numpy.random.RandomState(1234)
    #输入是一个矩阵，实际上矩阵的每一行都已是一个图像。
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10
    )

    #定义代价函数。用来进行梯度下降。
    cost = (classifier.negative_log_likelihood(y)
            + L1_reg * classifier.L1
            + L2_reg * classifier.L2
            )
            

    #定义梯度
    gparams = [T.grad(cost, param) for param in classifier.params]

    #更新梯度
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    #三个计算模型：训练，测试，验证。
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)

    # early-stopping参数
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    #保证每次迭代验证一次
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    #进行多次迭代
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        #每一次访问一个minibatch
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)

            iter = (epoch - 1) * n_train_batches + minibatch_index

            #对所有数据集合进行验证。求误差均值。
            validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
            this_validation_loss = numpy.mean(validation_losses)
            '''
                输出提示
            '''
            '''
            print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )
            '''

            #只有在验证的时候，有更好的效果才会保存。
            if (iter + 1) % validation_frequency == 0:
                #对所有数据集合进行验证。求误差均值。
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                #有更好的效果
                if this_validation_loss < best_validation_loss:
                    #如果改善的效果比较明显，则可以多训练几次。通过patience设置
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)
                    #记录当前的最佳结果的次数。
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    
                    #计算在测试模型上的分数
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    
                    '''
                        输出提示
                    '''
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            #终止条件，改善效率过低
            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    '''
        输出提示
    '''
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    test_mlp()