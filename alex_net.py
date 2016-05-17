import sys
sys.path.append('./lib')
import theano
theano.config.on_unused_input = 'warn'
import theano.tensor as T

import numpy as np

from layers import DataLayer, ConvPoolLayer, DropoutLayer, FCLayer, SoftmaxLayer


class AlexNet(object):

    def __init__(self, config):

        self.config = config

        batch_size = config['batch_size']
        flag_datalayer = config['use_data_layer']
        lib_conv = config['lib_conv']

        # ##################### BUILD NETWORK ##########################
        # allocate symbolic variables for the data
        # 'rand' is a random array used for random cropping/mirroring of data
        x = T.ftensor4('x')
        y = T.lvector('y')
        rand = T.fvector('rand')

        print '... building the model'
        self.layers = []
        params = []
        weight_types = []

        if flag_datalayer:
            data_layer = DataLayer(input=x, image_shape=(3, 256, 256,
                                                         batch_size),
                                   cropsize=227, rand=rand, mirror=True,
                                   flag_rand=config['rand_crop'])

            layer1_input = data_layer.output
        else:
            layer1_input = x

        convpool_layer1 = ConvPoolLayer(input=layer1_input,
                                        image_shape=(3, 227, 227, batch_size), 
                                        filter_shape=(3, 11, 11, 96), 
                                        convstride=4, padsize=0, group=1, 
                                        poolsize=3, poolstride=2, 
                                        bias_init=0.0, lrn=True,
                                        lib_conv=lib_conv,
                                        )
        self.layers.append(convpool_layer1)
        params += convpool_layer1.params
        weight_types += convpool_layer1.weight_type

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

        fc_layer6_input = T.flatten(
            convpool_layer5.output.dimshuffle(3, 0, 1, 2), 2)
        fc_layer6 = FCLayer(input=fc_layer6_input, n_in=9216, n_out=4096)
        self.layers.append(fc_layer6)
        params += fc_layer6.params
        weight_types += fc_layer6.weight_type

        dropout_layer6 = DropoutLayer(fc_layer6.output, n_in=4096, n_out=4096)

        fc_layer7 = FCLayer(input=dropout_layer6.output, n_in=4096, n_out=4096)
        self.layers.append(fc_layer7)
        params += fc_layer7.params
        weight_types += fc_layer7.weight_type

        dropout_layer7 = DropoutLayer(fc_layer7.output, n_in=4096, n_out=4096)

        softmax_layer8 = SoftmaxLayer(
            input=dropout_layer7.output, n_in=4096, n_out=1000)
        self.layers.append(softmax_layer8)
        params += softmax_layer8.params
        weight_types += softmax_layer8.weight_type

        # #################### NETWORK BUILT #######################

        self.cost = softmax_layer8.negative_log_likelihood(y)
        self.errors = softmax_layer8.errors(y)
        self.errors_top_5 = softmax_layer8.errors_top_x(y, 5)
        self.params = params
        self.x = x
        self.y = y
        self.rand = rand
        self.weight_types = weight_types
        self.batch_size = batch_size


