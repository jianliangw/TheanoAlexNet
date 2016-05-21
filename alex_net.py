import sys
sys.path.append('./lib')
import theano
theano.config.on_unused_input = 'warn'
import theano.tensor as T

import numpy as np

from layers import ConvPoolLayer, DropoutLayer, FCLayer, SoftmaxLayer


class AlexNet(object):

    def __init__(self, config):

        self.config = config

        batch_size = config['batch_size']
        lib_conv = config['lib_conv']
        useLayers = config['useLayers']
        imgWidth = config['imgWidth']
        imgHeight = config['imgHeight']
        initWeights = config['initWeights']  #if we wish to initialize alexnet with some weights. #need to make changes in layers.py to accept initilizing weights

        # ##################### BUILD NETWORK ##########################
        # allocate symbolic variables for the data
        # 'rand' is a random array used for random cropping/mirroring of data
        x = T.ftensor4('x')
        #y = T.lvector('y')

        print '... building the model'
        self.layers = []
        params = []
        weight_types = []

        if useLayers >= 1:
            convpool_layer1 = ConvPoolLayer(input=x,
                                        image_shape=(3, imgHeight, imgWidth, batch_size), 
                                        filter_shape=(3, 11, 11, 96), 
                                        convstride=4, padsize=0, group=1, 
                                        poolsize=3, poolstride=2, 
                                        bias_init=0.0, lrn=True,
                                        lib_conv=lib_conv,
                                        )
            self.layers.append(convpool_layer1)
            params += convpool_layer1.params
            weight_types += convpool_layer1.weight_type

        if useLayers >= 2:
            convpool_layer2 = ConvPoolLayer(input=convpool_layer1.output,
                                        image_shape=(96, 27, 27, batch_size),
                                        filter_shape=(96, 5, 5, 256), 
                                        convstride=1, padsize=2, group=2, 
                                        poolsize=3, poolstride=2, 
                                        bias_init=0.1, lrn=True,
                                        lib_conv=lib_conv,
                                        )
            self.layers.append(convpool_layer2)
            params += convpool_layer2.params
            weight_types += convpool_layer2.weight_type

        if useLayers >= 3:
            convpool_layer3 = ConvPoolLayer(input=convpool_layer2.output,
                                        image_shape=(256, 13, 13, batch_size),
                                        filter_shape=(256, 3, 3, 384), 
                                        convstride=1, padsize=1, group=1, 
                                        poolsize=1, poolstride=0, 
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        )
            self.layers.append(convpool_layer3)
            params += convpool_layer3.params
            weight_types += convpool_layer3.weight_type

        if useLayers >= 4:
            convpool_layer4 = ConvPoolLayer(input=convpool_layer3.output,
                                        image_shape=(384, 13, 13, batch_size),
                                        filter_shape=(384, 3, 3, 384), 
                                        convstride=1, padsize=1, group=2, 
                                        poolsize=1, poolstride=0, 
                                        bias_init=0.1, lrn=False,
                                        lib_conv=lib_conv,
                                        )
            self.layers.append(convpool_layer4)
            params += convpool_layer4.params
            weight_types += convpool_layer4.weight_type

        if useLayers >= 5:
            convpool_layer5 = ConvPoolLayer(input=convpool_layer4.output,
                                        image_shape=(384, 13, 13, batch_size),
                                        filter_shape=(384, 3, 3, 256), 
                                        convstride=1, padsize=1, group=2, 
                                        poolsize=3, poolstride=2, 
                                        bias_init=0.0, lrn=False,
                                        lib_conv=lib_conv,
                                        )
            self.layers.append(convpool_layer5)
            params += convpool_layer5.params
            weight_types += convpool_layer5.weight_type

        if useLayers >= 6:
            fc_layer6_input = T.flatten(convpool_layer5.output.dimshuffle(3, 0, 1, 2), 2)
            fc_layer6 = FCLayer(input=fc_layer6_input, n_in=9216, n_out=4096)
            self.layers.append(fc_layer6)
            params += fc_layer6.params
            weight_types += fc_layer6.weight_type
            dropout_layer6 = DropoutLayer(fc_layer6.output, n_in=4096, n_out=4096)

        if useLayers >= 7:
            fc_layer7 = FCLayer(input=dropout_layer6.output, n_in=4096, n_out=4096)
            self.layers.append(fc_layer7)
            params += fc_layer7.params
            weight_types += fc_layer7.weight_type
            dropout_layer7 = DropoutLayer(fc_layer7.output, n_in=4096, n_out=4096)

        if useLayers >= 8:
            softmax_layer8 = SoftmaxLayer(input=dropout_layer7.output, n_in=4096, n_out=1000)
            self.layers.append(softmax_layer8)
            params += softmax_layer8.params
            weight_types += softmax_layer8.weight_type

        # #################### NETWORK BUILT #######################

        self.output = self.layers[useLayers-1]
        self.params = params
        self.x = x
        #self.y = y
        self.weight_types = weight_types
        self.batch_size = batch_size
        self.useLayers = useLayers
        self.outLayer = self.layers[useLayers-1]

        self.forwardFunction = theano.function([self.x], [self.outLayer.output])
        
    def forward(self, inp):
        return self.forwardFunction(inp)
        
    #def train(self, updates, givens):
    #    self.trainModel = theano.function([], self.outLayer, updates=updates, givens=givens)
