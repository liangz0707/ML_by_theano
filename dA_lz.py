
# coding: utf-8

# #This tutorial introduces denoising auto-encoders (dA) using Theano

# 这篇文章是为了SdA的实现做准备的.
# 实际上dA和mlp都含有输入层、隐藏层、和输出层，唯一的区别是，dA使用了$W^T$作为了输出层的权重。这样做所隐含的意义：```是为了把输入x通过隐藏层变为y，再通过输出层还原成x```。通过这一组非线性变换可以学习到重要的主分量。
# 他和PCA的区别在于，dA是非线性的，而PCA是线性的。（由于增加了sigmod过程）

# 编码的过程为：
# $$y = f_{\theta}(x) = s(Wx+b)$$
# 解码的过程为：
# $$z = g_{\theta'}(y) = s(W'y + b')$$
# 值得范围：
# $$z,x \in [0,1]^d$$
# W可以进行约束：$W' = W^T$
# 梯度的误差可以写作：
# $$\sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]$$

# 参考文献 :
#    - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
#    Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
#    2008
#    - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
#    Training of Deep Networks, Advances in Neural Information Processing
#    Systems 19, 2007

# 引入必要的模块

# In[52]:

import os
import sys
import timeit
import cPickle
import gzip

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


# uitl这个单元并不是通过pip install安装的模块，而是一个在Deeplearing.net里面的py文件。具体可以在Github上搜索DeepLearningTutorials，使用里面的utils.py文件。
# 

# In[42]:

from utils import tile_raster_images


# In[43]:

try:
    import PIL.Image as Image
except ImportError:
    import Image


# 定义dA类
# $$y = s(W \tilde{x} + b)$$
# $$x = s(W' y  + b') $$
# $$L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]  $$
# W的随机初始值：
# $$ W\in (-4*sqrt(\frac{6.}{n_visible+n_hidden}) , 4*sqrt(\frac{6.}{n_hidden+n_visible}) $$

# In[44]:

class dA(object):
    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None
    ):
        #利用初始值进行赋值
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not W:
            #W如果没有给定则随机赋值
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)
        #bias值为0即可
        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        self.b = bhid
        self.b_prime = bvis
 
        self.W_prime = self.W.T
        #这也是一个随机量，但是他的生成和rng不同，它是专门提供shared的随机变量。
        self.theano_rng = theano_rng
        #记录输入的符号
        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input
        #保存要梯度下降的参数。
        self.params = [self.W, self.b, self.b_prime]
    #制造干扰参数，生成一组随机的干扰
    '''
        解释：如果只是用最小重构误差来进行约束的话，那么只不过是将输入映射成它本身而已。
        解决方法：1.使用稀疏，2.使用随机噪声（本文方法）
        解释：使用噪声是为了让隐藏层发现更多具有鲁棒性的特征。
    '''
    def get_corrupted_input(self, input, corruption_level):
        #这就是一个二项分布函数，让部分输入变成0
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input
    #得到隐藏层的值，也就是y
    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)
    #得到输出层的值，也就是由y还原的z
    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
    #误差更新，也就是梯度下降的过程
    def get_cost_updates(self, corruption_level, learning_rate):

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        #计算交叉熵
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)

        cost = T.mean(L)

        #计算梯度
        gparams = T.grad(cost, self.params)
        #生成梯度下降列表
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]
        #返回混杂熵，和梯度下降列表
        return (cost, updates)


# In[45]:

def test_dA(learning_rate=0.1, training_epochs=15,
            dataset='mnist.pkl.gz',
            batch_size=20, output_folder='dA_plots'):

    """
    同样是在手写数据集当中测试

    让后给出迭代次数 和每次使用的 训练样本数。

    """
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]

    #计算每次迭代的训练次数。
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    #索引作为输入，用于指向当前的训练batch
    index = T.lscalar() 
    #确定输入的类型
    x = T.matrix('x') 
    
    #结果文件建立
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    #os.chdir(output_folder)

    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=28 * 28,
        n_hidden=500
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = timeit.default_timer()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    print >> sys.stderr, ('The no corruption code for file ' +
                          ' ran for %.2fm' % ((training_time) / 60.))
    image = Image.fromarray(
        tile_raster_images(X=da.W.get_value(borrow=True).T,
                           img_shape=(28, 28), tile_shape=(10, 10),
                           tile_spacing=(1, 1)))
    image.save('filters_corruption_0.png')

    # start-snippet-3
    #####################################
    # BUILDING THE MODEL CORRUPTION 30% #
    #####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=28 * 28,
        n_hidden=500
    )
    #唯一给变的地方就是这里，修改了坍塌
    cost, updates = da.get_cost_updates(
        corruption_level=0.3,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = timeit.default_timer()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    print >> sys.stderr, ('The 30% corruption code for file ' +
                          ' ran for %.2fm' % (training_time / 60.))
    # end-snippet-3

    # start-snippet-4
    image = Image.fromarray(tile_raster_images(
        X=da.W.get_value(borrow=True).T,
        img_shape=(28, 28), tile_shape=(10, 20),
        tile_spacing=(1, 1)))
    image.save('filters_corruption_30.png')
    # end-snippet-4

    #os.chdir('../')


# In[50]:

def load_data(dataset):
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)

    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
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


# In[53]:

#test_dA()


# In[ ]:



