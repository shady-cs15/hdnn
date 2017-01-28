from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cPickle
from subprocess import call

import theano
import theano.tensor as T

import cv2

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

#image_name = '/home/satyaki/Desktop'
image_name = '/home/satyaki/Documents/HDNN/yt_cars/iidbUtgIi9OQaQ_img000269'
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

level_0_booleans = []
for i in range(2):
	cur_det_row = []
	for j in range(2):
		out_patch = out[:, :, i*64:(i+1)*64, j*64:(j+1)*64]
		if np.sum(out_patch)<1.:
			cur_det_row.append(False)
		else:
			cur_det_row.append(True)
	level_0_booleans.append(cur_det_row)
	
print level_0_booleans #remove
	
# read image at scale 256x256
# for booleans with level_0_booleans True 
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


# create model for level 2
load_file=open('trained_params_level_2_aerial.pkl', 'r')
params=()
for i in range(8):
	W = theano.shared(np.asarray(cPickle.load(load_file), dtype=theano.config.floatX), borrow=True)
	b = theano.shared(np.asarray(cPickle.load(load_file), dtype=theano.config.floatX), borrow=True)
	params+=([W, b], )
load_file.close()
dnn_level_2 = model(rng, input, (128, 128), batch_size=1, params=params, init=True)
h = theano.function([input], dnn_level_2.layer3.output)

# now read images to be fed to
# model dnn_layer_1
rgb_level_1_ = read_image(image_name, dims=(256, 256))
rgb_level_1 = rgb_level_1_.transpose(2, 0, 1).reshape(1, 3, 256, 256)
rgb_level_2_ = read_image(image_name, dims=(512, 512))
rgb_level_2 = rgb_level_2_.transpose(2, 0, 1).reshape(1, 3, 512, 512)

layer_1_outputs = []
level_1_booleans = []
for i in range(2):
	for j in range(2):
		if level_0_booleans[i][j]==False:
			for k in range(4):
				level_1_booleans.append(False)
			continue
		cur_patch = rgb_level_1[:, :, i*128:(i+1)*128, j*128:(j+1)*128]
		cur_output = g(cur_patch)
		threshold = (np.amax(cur_output) + np.amin(cur_output))/3.
		cur_output = (np.sign(cur_output - threshold)+1.)/2.
		for k in range(2):
			for l in range(2):
				out_patch = cur_output[:, :, k*64:(k+1)*64, l*64:(l+1)*64]
				if np.sum(out_patch)<1.:
					level_1_booleans.append(False)
				else:
					level_1_booleans.append(True)
		layer_1_outputs.append(cur_output)

print level_1_booleans


#threshold = (np.amax(np.asarray(layer_1_outputs)) + np.amin(np.asarray(layer_1_outputs)))/3.
level_2_heatmap = np.zeros((1, 1, 512, 512))
iter = 0
for i in range(2):
	for j in range(2):
		rgb_crop = rgb_level_2[:, :, i*256:(i+1)*256, j*256:(j+1)*256]
		temp_map = np.zeros((1, 1, 256, 256))
		for k in range(2):
			for l in range(2):
				if level_1_booleans[iter]==True:
					cur_patch = rgb_crop[:, :, k*128:(k+1)*128, l*128:(l+1)*128]
					cur_output = h(cur_patch)
					#threshold = (np.amax(cur_output) + np.amin(cur_output))/2.
					#cur_output = (np.sign(cur_output -threshold)+1.)/2.		
					temp_map[:, :, k*128:(k+1)*128, l*128:(l+1)*128]+=cur_output
				iter+=1
		level_2_heatmap[:, :, i*256:(i+1)*256, j*256:(j+1)*256]+=temp_map
			#cur_output = (np.sign(cur_output -threshold)+1.)/2.
			#for i in range(128):
			#	for j in range(128):
			#		cur_patch[0][2][i][j] = cur_patch[0][2][i][j]/2.0+ cur_output[0][0][i][j]/2.0
			#		cur_patch[0][1][i][j]/=2.0
			#		cur_patch[0][0][i][j]/=2.0
			#print iter
			#'''plt.subplot(2, 1, 2)
			#plt.axis('off')
			#plt.imshow(cur_output[0][0])
			#plt.subplot(2, 1, 1)
			#plt.axis('off')
			#plt.imshow(rgb_level_2_[i*128:(i+1)*128, j*128:(j+1)*128])
			#plt.show()'''
		#iter+=1

#plt.gray()
plt.subplot(1, 1, 1)
plt.axis('off')
plt.imshow(level_2_heatmap[0][0])
plt.show()

# snippet where bounding box is drawn
# assume 32x32 grid cells
# input image dimensions = 512x512

img = cv2.imread(image_name+'/rgb.jpg')
img = cv2.resize(img, (512, 512))
bin_threshold = (np.amax(level_2_heatmap[0][0]) + np.amin(level_2_heatmap[0][0]))/2.
bin_map = (np.sign(level_2_heatmap[0][0] - bin_threshold) +1.)/ 2. 

print bin_map.shape

plt.gray()
plt.subplot(1, 1, 1)
plt.axis('off')
plt.imshow(bin_map)
plt.show()

print img.shape

grid_booleans = []
sum_threshold = 3.
for i in range(16):
	for j in range(16):
		cur_grid_cell = bin_map[i*32:(i+1)*32, j*32:(j+1)*32]
		if np.sum(cur_grid_cell) > sum_threshold:
			grid_booleans.append(True)
			img[i*32:(i+1)*32, j*32:(j+1)*32] /= 2
			#img[i*32:(i+1)*32, j*32:(j+1)*32][2] += 128.
		else:
			grid_booleans.append(False)

cv2.imshow('frame', img)
cv2.waitKey(0)

#print grid_booleans

call(['mkdir', '/home/satyaki/Documents/HDNN/results/Desktop/'])#iidj114IJ1nwks_img000029/'])
plt.imsave('/home/satyaki/Documents/HDNN/results/Desktop/level2.png', level_2_heatmap[0][0])
