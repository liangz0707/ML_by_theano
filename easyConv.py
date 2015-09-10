# -*-  coding:utf-8 -*-
import theano
from theano import tensor as T
from theano.tensor.nnet import conv

import numpy

import pylab
from PIL import Image

rng = numpy.random.RandomState(2115)
#
input = T.tensor4()

w_shp = (2,3,9,9)
w_bound = numpy.sqrt(3*9*9)
W  = theano.shared(numpy.asarray(
        rng.uniform(
            low = -1.0/w_bound,
            high= 1.0/w_bound,
            size = w_shp
            ),
        dtype=input.dtype)
        )
b_shp = (2,)
#b=theano.shared(numpy.asarray(
#        rng.uniform(low=.5, high=.5,size =b_shp),dtype=input.dtype
#        ))
b=theano.shared(numpy.ones(b_shp,dtype=input.dtype))
#这两个参数的第一维所指的含义不同，input表示的是batch的个数，W表示的是滤波器的个数。得到的结果是batch个图像，每个产生filter_size组所以结果为 （batch,filter_size,h,w)
image_shape = (1,3,300,400)
conv_out = conv.conv2d(input,W,
            filter_shape=w_shp,
            image_shape=image_shape)
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))
f = theano.function([input], output)


img = Image.open('3wolfmoon.jpg')
img = numpy.asarray(img, dtype='float32') / 256.
img_ = img.transpose(2, 0, 1).reshape(1, 3, 639, 516)
filtered_img = f(img_);
print(filtered_img[0][1].shape)
print(filtered_img[0][0].shape)

# plot original image and first and second components of output
pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray();
# recall that the convOp output (filtered image) is actually a "minibatch",
# of size 1 here, so we take index 0 in the first dimension:
#表示第1个batch的第1个滤波器的结果
pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
#表示第1个batch的第2个滤波器的结果
pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])
pylab.show()