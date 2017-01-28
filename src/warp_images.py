from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cPickle
from subprocess import call

#import theano
#import theano.tensor as T

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

dirs = ['iidj114IJ1nwks_img000029', 'iid105542012_img001053', 'iid87IdtQjL5Ng_img000759', 'iid8meH0vZt08k_img000350']
for dir_name in dirs:
	image_name = '/home/satyaki/Documents/HDNN/yt_cars/'+dir_name
	rgb = cv2.imread(image_name+'/rgb.jpg')
	rgb = cv2.resize(rgb, (512, 512))
	cv2.imshow('rgb', rgb)
	cv2.waitKey(0)
	cv2.imwrite('/home/satyaki/Documents/HDNN/results/'+dir_name+'/rgb.jpg', rgb)

#plt.subplot(1, 1, 1)
#plt.axis('off')
#plt.imshow(rgb)
#plt.show()

#plt.imsave('/home/satyaki/Documents/HDNN/results/'+dir_name+'/rgb.jpg', rgb)