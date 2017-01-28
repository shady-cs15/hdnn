from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cPickle

import theano
import theano.tensor as T


from model import model

def read_image(dir_name, type='rgb'):
	img = None
	if type=='rgb':
		img = Image.open(open(dir_name+'/rgb.jpg'))
	else:
		img = Image.open(open(dir_name+'/blend.jpg')) #not blend check name of file
	img = img.resize((128, 128), Image.ANTIALIAS)
	img = np.asarray(img, dtype=theano.config.floatX)
	return img

dir_name = 'iid94708919_img000932'
rgb = read_image('/home/satyaki/Documents/HDNN/yt_cars/'+dir_name)
rgb_ = rgb.transpose(2, 0, 1).reshape(1, 3, 128, 128)

load_file=open('trained_params_level_0_aerial.pkl', 'r')
params=()
for i in range(8):
	W = theano.shared(np.asarray(cPickle.load(load_file), dtype=theano.config.floatX), borrow=True)
	b = theano.shared(np.asarray(cPickle.load(load_file), dtype=theano.config.floatX), borrow=True)
	params+=([W, b], )
load_file.close()

input=T.tensor4()
rng = np.random.RandomState(23455)
dnn = model(rng, input, (128, 128), batch_size=1, params=params, init=True)
f = theano.function([input], dnn.layer4.output)

out = f(rgb_)

print np.amax(out), np.amin(out)
threshold = (np.amax(out) + np.amin(out))/3.
#out = (np.sign(out - threshold)+1.)/2.

print np.amax(out), np.amin(out)

plt.gray()
plt.subplot(1, 1, 1)
plt.axis('off')
plt.imshow(out[0][0])
plt.show()

#plt.imsave('/home/satyaki/Documents/HDNN/results/'+dir_name+'/level0.png', out[0][0])