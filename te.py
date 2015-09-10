import numpy
import theano
import theano.tensor as T

x = T.dscalar()
y = T.dscalar()
index = T.dscalar()
w = theano.shared(numpy.ones(3,),theano.config.floatX)
result = w*x
c = theano.function([x],result)
d= theano.function(
        inputs=[index],
        outputs=result,
        givens={
            x:1.0+index
        }
    )
print(c(1))
print(d(1))