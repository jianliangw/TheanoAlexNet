#import sys
#sys.path.append('./lib')
from alex_net import AlexNet, compile_models
import yaml
import scipy.misc
import matplotlib.pyplot as plt


with open('config.yaml', 'r') as f:
    config = yaml.load(f)
img = scipy.misc.imread('cat.jpg')
print img.shape   #(360, 480, 3)  : height, width, channel
img = scipy.misc.imresize(img, (config['imgHeight'], config['imgWidth']))

#plt.imshow(img)
#plt.show()

alexnetModel = AlexNet(config)
(train_model, validate_model, train_error, learning_rate, shared_x, shared_y, rand_arr, vels) = compile_models(alexnetModel, config)