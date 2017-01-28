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
h = theano.function([input], dnn_level_2.layer8.output)


file = open('testdirs.txt', 'r')

dir_list = []
for line in file:
	dir_list.append(line)
file.close()

for i in range(len(dir_list)-1):
	dir_list[i] = dir_list[i][:-1]

print '\n'
p = 0
for sum_threshold in range(100, 1000, 50):
	false_positives = 0
	false_negatives = 0
	true_positives = 0
	true_negatives = 0
	for dir in dir_list:
		p += 1
		print '\033[F ', p, '/', len(dir_list)
		image_name = '/home/satyaki/Documents/HDNN/yt_cars/'+dir
		rgb = read_image(image_name)
		rgb_ = rgb.transpose(2, 0, 1).reshape(1, 3, 128, 128)

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
		#print level_0_booleans

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

		#print level_1_booleans

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
							temp_map[:, :, k*128:(k+1)*128, l*128:(l+1)*128]+=cur_output
						iter+=1
				level_2_heatmap[:, :, i*256:(i+1)*256, j*256:(j+1)*256]+=temp_map

		label = cv2.imread(image_name+'/labels_256x256.jpg', cv2.IMREAD_GRAYSCALE)
		label = cv2.resize(label, (512, 512))
		
		img = cv2.imread(image_name+'/rgb.jpg')
		img = cv2.resize(img, (512, 512))
		bin_threshold = 0.4 #(np.amax(level_2_heatmap[0][0]) + np.amin(level_2_heatmap[0][0]))/3.
		bin_map = (np.sign(level_2_heatmap[0][0] - bin_threshold) +1.)/ 2. 

		grid_booleans = []
		label_booleans = []
		#sum_threshold = 100#3
		for i in range(16):
			for j in range(16):
				cur_grid_cell = bin_map[i*32:(i+1)*32, j*32:(j+1)*32]
				cur_label_cell = label[i*32:(i+1)*32, j*32:(j+1)*32]
				if np.sum(cur_label_cell) > sum_threshold:
					label_booleans.append(True)
				else:
					label_booleans.append(False)
				if np.sum(cur_grid_cell) > sum_threshold:
					grid_booleans.append(True)
				else:
					grid_booleans.append(False)

		for i in range(len(grid_booleans)):
			if grid_booleans[i]==True and label_booleans[i]==True:
				true_positives += 1
			if grid_booleans[i]==False and label_booleans[i]==False:
				true_negatives += 1
			if grid_booleans[i]==False and label_booleans[i]==True:
				false_negatives += 1
			if grid_booleans[i]==True and label_booleans[i]==False:
				false_positives += 1

	total = false_positives + false_negatives + true_negatives + true_positives
	print 'false +ves =', false_positives*100./total, '%'
	print 'false -ves =', false_negatives*100./total, '%'
	print 'true +ves =', true_positives*100./total, '%'
	print 'true -ves =', true_negatives*100./total, '%'
	print 'precision =', true_positives*1./(true_positives+false_positives)
	print 'recall =', true_positives*1./(true_positives+false_negatives)
	print 'threshold =', sum_threshold
	print '\n'