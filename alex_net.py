import sys
sys.path.append('./lib')
import theano
from theano import In
theano.config.on_unused_input = 'warn'
import theano.tensor as T

import numpy as np

from layers import ConvPoolLayer, DropoutLayer, FCLayer, SoftmaxLayer


class AlexNet(object):
#todo: add mean file subtraction
#todo: change the fixed sizes of conv layer inputs, eg in layer 2 its 27x27
#todo: #x is 4d, with 'batch' number of images. meanVal has only '1' in the 'batch' dimension. subtraction wont work
#todo: batch size.
#todo: switch off drop off during testing. also allow drop out ratio to be set in config file
    def __init__(self, config):

        self.config = config

        batch_size = config['batch_size']
        lib_conv = config['lib_conv']
        useLayers = config['useLayers']
        imgWidth = config['imgWidth']
        imgHeight = config['imgHeight']
        initWeights = config['initWeights']  #if we wish to initialize alexnet with some weights. #need to make changes in layers.py to accept initilizing weights
        if initWeights:
            weightsDir = config['weightsDir']
            weightFileTag = config['weightFileTag']

        # ##################### BUILD NETWORK ##########################
        # allocate symbolic variables for the data
        # 'rand' is a random array used for random cropping/mirroring of data
        x = T.ftensor4('x')
        mean = T.ftensor4('mean')
        #y = T.lvector('y')

        print '... building the model'
        self.layers = []
        params = []
        weight_types = []

        if useLayers >= 1:
            convpool_layer1 = ConvPoolLayer(input=x-mean,
                                        image_shape=(3, imgHeight, imgWidth, batch_size), 
                                        filter_shape=(3, 11, 11, 96), 
                                        convstride=4, padsize=0, group=1, 
                                        poolsize=3, poolstride=2, 
                                        bias_init=0.0, lrn=True,
                                        lib_conv=lib_conv,
                                        initWeights=initWeights, weightsDir=weightsDir, weightFiles=['W_0'+weightFileTag, 'b_0'+weightFileTag]
                                        )
            self.layers.append(convpool_layer1)
            params += convpool_layer1.params
            weight_types += convpool_layer1.weight_type

        if useLayers >= 2:
            convpool_layer2 = ConvPoolLayer(input=convpool_layer1.output,
                                        image_shape=(96, 27, 27, batch_size),    #change from 27 to appropriate value sbased on conv1's output
                                        filter_shape=(96, 5, 5, 256), 
                                        convstride=1, padsize=2, group=2, 
                                        poolsize=3, poolstride=2, 
                                        bias_init=0.1, lrn=True,
                                        lib_conv=lib_conv,
                                        initWeights=initWeights, weightsDir=weightsDir, weightFiles=['W0_1'+weightFileTag, 'W1_1'+weightFileTag, 'b0_1'+weightFileTag, 'b1_1'+weightFileTag]
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
                                        initWeights=initWeights, weightsDir=weightsDir, weightFiles=['W_2'+weightFileTag, 'b_2'+weightFileTag]
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
                                        initWeights=initWeights, weightsDir=weightsDir, weightFiles=['W0_3'+weightFileTag, 'W1_3'+weightFileTag, 'b0_3'+weightFileTag, 'b1_3'+weightFileTag]
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
                                        initWeights=initWeights, weightsDir=weightsDir, weightFiles=['W0_4'+weightFileTag, 'W1_4'+weightFileTag, 'b0_4'+weightFileTag, 'b1_4'+weightFileTag]
                                        )
            self.layers.append(convpool_layer5)
            params += convpool_layer5.params
            weight_types += convpool_layer5.weight_type

        if useLayers >= 6:
            fc_layer6_input = T.flatten(convpool_layer5.output.dimshuffle(3, 0, 1, 2), 2)
            fc_layer6 = FCLayer(input=fc_layer6_input, n_in=9216, n_out=4096, initWeights=initWeights, weightsDir=weightsDir, weightFiles=['W_5'+weightFileTag, 'b_5'+weightFileTag])
            self.layers.append(fc_layer6)
            params += fc_layer6.params
            weight_types += fc_layer6.weight_type
            dropout_layer6 = DropoutLayer(fc_layer6.output, n_in=4096, n_out=4096)

        if useLayers >= 7:
            fc_layer7 = FCLayer(input=dropout_layer6.output, n_in=4096, n_out=4096, initWeights=initWeights, weightsDir=weightsDir, weightFiles=['W_6'+weightFileTag, 'b_6'+weightFileTag])
            self.layers.append(fc_layer7)
            params += fc_layer7.params
            weight_types += fc_layer7.weight_type
            dropout_layer7 = DropoutLayer(fc_layer7.output, n_in=4096, n_out=4096)

        if useLayers >= 8:
            softmax_layer8 = SoftmaxLayer(input=dropout_layer7.output, n_in=4096, n_out=1000, initWeights=initWeights, weightsDir=weightsDir, weightFiles=['W_7'+weightFileTag, 'b_7'+weightFileTag])
            self.layers.append(softmax_layer8)
            params += softmax_layer8.params
            weight_types += softmax_layer8.weight_type

        # #################### NETWORK BUILT #######################

        self.output = self.layers[useLayers-1]
        self.params = params
        self.x = x
        self.mean = mean
        #self.y = y
        self.weight_types = weight_types
        self.batch_size = batch_size
        self.useLayers = useLayers
        self.outLayer = self.layers[useLayers-1]

        meanVal = np.load(config['mean_file'])
        meanVal = meanVal[:, :, :, np.newaxis].astype('float32')   #x is 4d, with 'batch' number of images. meanVal has only '1' in the 'batch' dimension. subtraction wont work.
        #print meanVal.shape, 'ggggggg'

        if useLayers >= 8:  #if last layer is softmax, then its output is y_pred
            finalOut = self.outLayer.y_pred
        else:
            finalOut = self.outLayer.output
        self.forwardFunction = theano.function([self.x, In(self.mean, value=meanVal)], [finalOut])
        
    def forward(self, inp):
        if len(inp.shape) == 3:  #if a single image was input, convert it into 4d form
            inp = np.rollaxis(inp, 2)
            inp = inp[:, :, :, np.newaxis].astype('float32')
            #print inp.shape, 'ffffffff'
        #print inp.shape, 'ffffffff'
        return self.forwardFunction(inp)
        
    #def train(self, updates, givens):
    #    self.trainModel = theano.function([], self.outLayer, updates=updates, givens=givens)
