#import sys
#sys.path.append('./lib')
from alex_net import AlexNet
import yaml
import scipy.misc
import matplotlib.pyplot as plt


#THEANO_FLAGS='mode=FAST_COMPILE' python testAlexNet.py 

with open('config.yaml', 'r') as f:
    config = yaml.load(f)
img = scipy.misc.imread('cat.jpg')
print img.shape   #(360, 480, 3)  : height, width, channel
img = scipy.misc.imresize(img, (config['imgHeight'], config['imgWidth']))

#plt.imshow(img)
#plt.show()



alexnetModel = AlexNet(config)
print alexnetModel.forward(img)

print 'done'


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
