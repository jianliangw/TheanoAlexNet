# TheanoAlexNet
*************************
summary
*************************
Deploying Alexnet using Theano

This code is built by modifying this
https://github.com/uoguelph-mlrg/theano_alexnet/tree/master/lib


The alexnet implementation in the above link trains a full alexnet from scratch. However, sometimes, we need to use pre-trained alexnet in a transferred learning type or finetuning scenario, or maybe use only the first few layers of alexnet.
This codebase tries to simplify those use-cases of alexnet.

I have modified the code from U. of Guelph's alexnet, by removing their cost funcion, training mechanism, data layer etc. This alexnet only implements a forward pass neural network. One can use this as is or use it in conjuction with a cost function and training mechanism defined outside for finetuning etc.


*************************
Files description
*************************
testAlexNet.py: a small file showing how to call and use alex_net.py.

alex_net.py: the main file that defines the alexnet class. The class takes 2 inputs, a config (read from config.yaml) and a train/test mode indicator.
self.params and self.output are the two important things one may need for training etc.
Also forward() function is useful in test mode for forward propogation.

config.yaml: configurations. The parameters in the file are described below:
useLayers: the number of layers to be used. If you set useLayers <= 5, then only the convolutional layers are in action (1 to 5). So in that case input size can be arbitrary. If useLayers is 6, 7 8 or greater, then fully connected layers are used, in which the input must be 227x227, else it will fail.
imgWidth, imgHeight: the dimensions to which the input image is resized once it is read.
batch_size: the batchsize. During input, one can specify <= batch_size number of images at one go.
initWeights: load pretrained weights if it is True, else initialize weights randomly
weightsDir: directory containing the pretrained weights
mean_file: location of mean file
prob_drop: probability to be used in dropout layer
weightFileTag: set to '_65' if you are using U of Guelph's pretrained weights
parameters: a folder containing pretrained weights. For pretrained weights, see download instructions here: https://github.com/uoguelph-mlrg/theano_alexnet/tree/master/pretrained/alexnet

lib: layers.py is the file that contains layer definitions




