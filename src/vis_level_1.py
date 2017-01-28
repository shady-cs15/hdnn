from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cPickle
from subprocess import call

import theano
import theano.tensor as T


from model import model

def read_image(dir_name, type='rgb', dims=(128, 128)):
	img = None
	if type=='rgb':
		img = Image.open(open(dir_name+'/rgb.jpg'))
	else:
		img = Image.open(open(dir_name+'/blend.jpg')) #not blend check name of file
	img = img.resize(dims, Image.ANTIALIAS)
	img = np.asarray(img, dtype=theano.config.floatX)
	return img

image_name = '/home/satyaki/Desktop'
#image_name = '/home/satyaki/Documents/HDNN/yt_cars/iidj114IJ1nwks_img000029'
rgb = read_image(image_name)
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
f = theano.function([input], dnn.layer8.output)

out_ = f(rgb_)
threshold = (np.amax(out_) + np.amin(out_))/3.
out = (np.sign(out_ - threshold)+1.)/2.

level_0_quadrants = []
#level_0_preds = []
for i in range(2):
	cur_det_row = []
	#cur_preds_row = []
	for j in range(2):
		out_patch = out[:, :, i*64:(i+1)*64, j*64:(j+1)*64]
		#out_patch_resized = out_patch.resize((128, 128), Image.ANTIALIAS)
		#cur_preds_row.append(out_patch_resized)
		if np.sum(out_patch)<1.:
			cur_det_row.append(False)
		else:
			cur_det_row.append(True)
		#plt.gray()
		#plt.subplot(1, 1, 1)
		#plt.axis('off')
		#plt.imshow(out_patch_resized)
		#plt.show()
	level_0_quadrants.append(cur_det_row)
	#level_0_preds.append(cur_preds_row)

print level_0_quadrants #remove
	
# read image at scale 256x256
# for quadrants with level_0_quadrant True 
# predict output
# first create model for level 1
load_file=open('trained_params_level_1_aerial.pkl', 'r')
params=()
for i in range(8):
	W = theano.shared(np.asarray(cPickle.load(load_file), dtype=theano.config.floatX), borrow=True)
	b = theano.shared(np.asarray(cPickle.load(load_file), dtype=theano.config.floatX), borrow=True)
	params+=([W, b], )
load_file.close()
dnn_level_1 = model(rng, input, (128, 128), batch_size=1, params=params, init=True)
g = theano.function([input], dnn_level_1.layer8.output)

# now read images to be fed to
# model dnn_layer_1
rgb_level_1_ = read_image(image_name, dims=(256, 256))
rgb_level_1 = rgb_level_1_.transpose(2, 0, 1).reshape(1, 3, 256, 256)
level_1_outputs = []
for i in range(2):
	for j in range(2):
		if level_0_quadrants[i][j]==False:
			continue
		cur_patch = rgb_level_1[:, :, i*128:(i+1)*128, j*128:(j+1)*128]
		cur_output = g(cur_patch)
		level_1_outputs.append(cur_output)

level_1_map = np.zeros((1, 1, 256, 256))
threshold = (np.amax(np.asarray(level_1_outputs)) + np.amin(np.asarray(level_1_outputs)))/3.
iter = 0
for i in range(2):
	for j in range(2):
		if level_0_quadrants[i][j]==False:
			continue
		cur_output = level_1_outputs[iter]
		iter+=1
		level_1_map[:, :, i*128:(i+1)*128, j*128:(j+1)*128]+=cur_output
		# remove from here
		#cur_output = (np.sign(cur_output - threshold)+1.)/2.
		#plt.gray()
		
plt.subplot(1, 1, 1)
plt.axis('off')
plt.imshow(level_1_map[0][0])
plt.show()
		

#plt.gray()
#plt.subplot(1, 1, 1)
#plt.axis('off')
#plt.imshow(out[0][0])
#plt.show()

call(['mkdir', '/home/satyaki/Documents/HDNN/results/Desktop/'])#iidj114IJ1nwks_img000029/'])
plt.imsave('/home/satyaki/Documents/HDNN/results/Desktop/level1.png', level_1_map[0][0])
