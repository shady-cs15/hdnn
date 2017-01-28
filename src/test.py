from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

import theano
import theano.tensor as T

from conv_layer import conv_pool_layer
from deconv_layer import deconv_unpool_layer
from model import model


def read_image(dir_name, type='rgb'):
	img = None
	if type=='rgb':
		img = Image.open(open(dir_name+'/rgb.jpg'))
	else:
		img = Image.open(open(dir_name+'/labels.png'))
	img = img.resize((256, 256), Image.ANTIALIAS)
	img = np.asarray(img, dtype='float64')
	return img

rgb = read_image('../yt_cars')/256.
rgb_ = rgb.transpose(2, 0, 1).reshape(1, 3, 128, 128)
label = read_image('../yt_cars', 'label').reshape(1, 1, 128, 128)

plt.gray()
plt.subplot(1, 1, 1)
plt.axis('off')
plt.imshow(label[0][0])
plt.show()
plt.imsave("label_.png", label[0][0])

#test_img = theano.shared(np.array(rgb, dtype=theano.config.floatX), borrow=True)
input = T.tensor4()
output = T.tensor4()
index = T.scalar()
rng = np.random.RandomState(23455)
dummy_wt = theano.shared(np.asarray(rng.uniform(low=-1., high=-1., size=(1, 1)), dtype=theano.config.floatX), borrow=True)
params = ([dummy_wt, ]*2, )*20
dnn = model(rng, input, (128, 128), batch_size=1, params=params)
#f = theano.function([input], dnn.layer20.output)
'''
out = f(rgb_)
plt.subplot(1, 1, 1)
plt.axis('off')
plt.imshow(out[0, 0, :, :])
plt.show()
'''

'''
# sample training
learning_rate=0.001
cost = T.mean(T.sqr(dnn.layer20.output-output))
params = dnn.layer1.params + \
			dnn.layer2.params + \
			dnn.layer3.params + \
			dnn.layer4.params + \
			dnn.layer5.params + \
			dnn.layer6.params + \
			dnn.layer7.params + \
			dnn.layer8.params + \
			dnn.layer9.params + \
			dnn.layer10.params + \
			dnn.layer11.params + \
			dnn.layer12.params + \
			dnn.layer13.params + \
			dnn.layer14.params + \
			dnn.layer15.params + \
			dnn.layer16.params + \
			dnn.layer17.params + \
			dnn.layer18.params + \
			dnn.layer19.params + \
			dnn.layer20.params

grads = T.grad(cost, params)
updates = [
	(param_i, param_i - learning_rate*grad_i)
	for param_i, grad_i in zip(params, grads)
]

train_img = theano.shared(rgb_, borrow=True)
train_label = theano.shared(label, borrow=True)
train_fn = theano.function([], cost, updates=updates,
								givens={input:train_img,
										output:train_label})

print 'landmark'

n_epochs=50
epoch=0
while(epoch < n_epochs):
	cost_ij = train_fn()
	print 'epoch :', epoch, ' cost: ', cost_ij
	epoch+=1


dnn_viz = model(rng, input, (128, 128), batch_size=1, params=params)
f = theano.function([input], dnn_viz.layer20.output)
out = f(rgb_)
plt.subplot(1, 1, 1)
plt.axis('off')
plt.imshow(out[0, 0, :, :])
plt.show()
'''
