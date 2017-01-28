import cv2
import os
from PIL import Image
import numpy as np

img_dim = 512
dir_names = []
for f in os.listdir('../yt_cars/'):#[0]:
	#f = 'iidfCwKkQ_rX8c_img000842'
	label = cv2.imread('../yt_cars/'+f+'/labels_viz.png', 0)
	img = cv2.imread('../yt_cars/'+f+'/rgb.jpg')
	if label.shape[0]<=0 or label.shape[1]<=0:
		print f
	label = cv2.resize(label, (img_dim, img_dim))
	img = cv2.resize(img, (img_dim, img_dim))
	
	bin_, thresh1 = cv2.threshold(label, 60, 255,cv2.THRESH_BINARY)
	dilated = cv2.dilate(thresh1, np.ones((3, 3), np.uint8)) #'''(5*img_dim/128, 5*img_dim/128)''', np.uint8))
	blurred = cv2.GaussianBlur(dilated, (11, 11), 5)
	blended = img
	
	for i in range(img_dim):
		for j in range(img_dim):
			blended[i][j][2]= blended[i][j][2]/2.0+ blurred[i][j]/2.0
			blended[i][j][1]/=2.0
			blended[i][j][0]/=2.0
	blended = cv2.resize(blended, (512, 512))

	cv2.imshow('label', label)
	cv2.imshow('binary', thresh1)
	cv2.imshow('dilated', dilated)
	cv2.imshow('blurred', blurred)
	cv2.imshow('blended', blended)
	cv2.imwrite('../yt_cars/'+f+'/labels_512x512.jpg', dilated)
	k = cv2.waitKey(33)
	if k==27:
		break

 
