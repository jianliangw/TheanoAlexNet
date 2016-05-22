#import sys
#sys.path.append('./lib')
from alex_net import AlexNet
import yaml
#import scipy.misc
#import numpy as np


#THEANO_FLAGS='mode=FAST_COMPILE' python testAlexNet.py 

with open('config.yaml', 'r') as f:
    config = yaml.load(f)


alexnetModel = AlexNet(config, True)

"""
x = alexnetModel.forward(['cat.jpg'])
print x[0].shape
print type(x[0])
print 'done 1'
"""


x = alexnetModel.forward(['cat.jpg','cat.jpg'])
print x[0].shape
print type(x[0])
print 'done 2'
#y = alexnetModel.forward(['cat.jpg','cat.jpg'])
#print x[0] == y[0]
#print x[0]
#print y[0]


"""
import theano.tensor as T
import theano, numpy as np

params = theano.shared(value=np.cast[theano.config.floatX](2 * np.ones((1,2), dtype=theano.config.floatX)))
x = T.dvector()
f = T.sum(T.dot(params, x))
print f
g = T.grad(f, x)
print g
fprime = theano.function([x], g)
print fprime([1,2])
"""
