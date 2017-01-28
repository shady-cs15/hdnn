from conv_layer import conv_pool_layer
from deconv_layer import deconv_unpool_layer

import theano
import theano.tensor as T

import numpy as np

class model(object):
	def __init__(self, rng, input, input_dim, batch_size=5, init=False, params=None):

		# assume input dimensions is 128x128
		self.input = input
		self.inp_h = input_dim[0]
		self.inp_w = input_dim[1]

		self.layer1 = conv_pool_layer(
			rng,
			input = input,
			image_shape=(batch_size, 3, self.inp_h, self.inp_w),
			filter_shape=(64, 3, 3, 3),
			poolsize=(2, 2),
			zero_pad=True,
			read_file=init,
			W_input=params[0][0],
			b_input=params[0][1]
		)

		# input is 64x64x64
		self.layer2 = conv_pool_layer(
			rng,
			input = self.layer1.output,
			image_shape=(batch_size, 64, self.inp_h/2, self.inp_w/2),
			filter_shape=(128, 64, 3, 3),
			poolsize=(2, 2),
			zero_pad=True,
			read_file=init,
			W_input=params[1][0],
			b_input=params[1][1]
		)

		# input is 128x32x32
		self.layer3 = conv_pool_layer(
			rng,
			input = self.layer2.output,
			image_shape=(batch_size, 128, self.inp_h/4, self.inp_w/4),
			filter_shape=(512, 128, 3, 3),
			poolsize=(2, 2),
			zero_pad=True,
			read_file=init,
			W_input=params[2][0],
			b_input=params[2][1]
		)

		# input is 512x16x16
		self.layer4 = conv_pool_layer(
			rng,
			input = self.layer3.output,
			image_shape=(batch_size, 512, self.inp_h/8, self.inp_w/8),
			filter_shape=(1024, 512, 3, 3),
			poolsize=(2, 2),
			zero_pad=True,
			read_file=init,
			W_input=params[3][0],
			b_input=params[3][1]
		)

		# decovolution starts here
		# input is 1024x8x8
		self.layer5 = deconv_unpool_layer(
			rng,
			input = self.layer4.output,
			image_shape=(batch_size, 1024, self.inp_h/16, self.inp_w/16),
			filter_shape=(512, 1024, 3, 3),
			zero_pad=True,
			non_linearity=True,
			switch=self.layer4.switch,
			unpoolsize=(2, 2),
			read_file=init,
			W_input=params[4][0],
			b_input=params[4][1]
		)

		self.layer6 = deconv_unpool_layer(
			rng,
			input = self.layer5.output,
			image_shape=(batch_size, 512, self.inp_h/8, self.inp_w/8),
			filter_shape=(128, 512, 3, 3),
			zero_pad=True,
			non_linearity=True,
			switch=self.layer3.switch,
			unpoolsize=(2, 2),
			read_file=init,
			W_input=params[5][0],
			b_input=params[5][1]
		)
		
		self.layer7 = deconv_unpool_layer(
			rng,
			input = self.layer6.output,
			image_shape=(batch_size, 128, self.inp_h/4, self.inp_w/4),
			filter_shape=(64, 128, 3, 3),
			zero_pad=True,
			non_linearity=True,
			switch=self.layer2.switch,
			unpoolsize=(2, 2),
			read_file=init,
			W_input=params[6][0],
			b_input=params[6][1]
		)

		self.layer8 = deconv_unpool_layer(
			rng,
			input = self.layer7.output,
			image_shape=(batch_size, 64, self.inp_h/2, self.inp_w/2),
			filter_shape=(1, 64, 3, 3),
			zero_pad=True,
			non_linearity=True,
			sigmoid=True,
			switch=self.layer1.switch,
			unpoolsize=(2, 2),
			read_file=init,
			W_input=params[7][0],
			b_input=params[7][1]
		)

		'''
		# input is 16x32x32
		self.layer5 = conv_pool_layer(
			rng,
			input = self.layer4.output,
			image_shape=(batch_size, 16, self.inp_h/4, self.inp_w/4),
			filter_shape=(32, 16, 3, 3),
			poolsize=(1, 1),
			zero_pad=True,
			read_file=init,
			W_input=params[4][0],
			b_input=params[4][1]
		)

		# input is 32x32x32
		self.layer6 = conv_pool_layer(
			rng,
			input = self.layer5.output,
			image_shape=(batch_size, 32, self.inp_h/4, self.inp_w/4),
			filter_shape=(64, 32, 3, 3),
			poolsize=(2, 2),
			zero_pad=True,
			read_file=init,
			W_input=params[5][0],
			b_input=params[5][1]
		)

		# input is 64x16x16
		self.layer7 = conv_pool_layer(
			rng,
			input = self.layer6.output,
			image_shape=(batch_size, 64, self.inp_h/8, self.inp_w/8),
			filter_shape=(128, 64, 3, 3),
			poolsize=(2, 2),
			zero_pad=True,
			read_file=init,
			W_input=params[6][0],
			b_input=params[6][1]
		)

		# input is 128x8x8
		self.layer8 = conv_pool_layer(
			rng,
			input = self.layer7.output,
			image_shape=(batch_size, 128, self.inp_h/16, self.inp_w/16),
			filter_shape=(256, 128, 3, 3),
			poolsize=(2, 2),
			read_file=init,
			zero_pad=True,
			W_input=params[7][0],
			b_input=params[7][1]
		)

		# input is 256x4x4
		self.layer9 = conv_pool_layer(
			rng,
			input = self.layer8.output,
			image_shape=(batch_size, 256, self.inp_h/32, self.inp_w/32),
			filter_shape=(512, 256, 3, 3),
			poolsize=(2, 2),
			read_file=init,
			zero_pad=True,
			W_input=params[8][0],
			b_input=params[8][1]
		)

		# input is 512x2x2
		self.layer10 = conv_pool_layer(
			rng,
			input = self.layer9.output,
			image_shape=(batch_size, 512, self.inp_h/64, self.inp_w/64),
			filter_shape=(1024, 512, 3, 3),
			poolsize=(2, 2),
			read_file=init,
			zero_pad=True,
			W_input=params[9][0],
			b_input=params[9][1]
		)

		# input is 1024x1x1
		# deconvolution unpooling begins from here
		self.layer11 = deconv_unpool_layer(
			rng,
			input = self.layer10.output,
			image_shape=(batch_size, 1024, self.inp_h/128, self.inp_w/128),
			filter_shape=(512, 1024, 3, 3),
			zero_pad=True,
			non_linearity=True,
			switch=self.layer10.switch,
			unpoolsize=(2, 2),
			read_file=init,
			W_input=params[10][0],
			b_input=params[10][1]
		)

		# input is 512x2x2
		self.layer12 = deconv_unpool_layer(
			rng,
			input = self.layer11.output,
			image_shape=(batch_size, 512, self.inp_h/64, self.inp_w/64),
			filter_shape=(256, 512, 3, 3),
			zero_pad=True,
			non_linearity=True,
			switch=self.layer9.switch,
			unpoolsize=(2, 2),
			read_file=init,
			W_input=params[11][0],
			b_input=params[11][1]
		)

		# input is 256x4x4
		self.layer13 = deconv_unpool_layer(
			rng, 
			input=self.layer12.output,
			image_shape=(batch_size, 256, self.inp_h/32, self.inp_w/32),
			filter_shape=(128, 256, 3, 3),
			zero_pad=True,
			non_linearity=True,
			switch=self.layer8.switch,
			unpoolsize=(2, 2),
			read_file=init,
			W_input=params[12][0],
			b_input=params[12][1]
		)

		# input is 128x8x8
		self.layer14 = deconv_unpool_layer(
			rng,
			input=self.layer13.output,
			image_shape=(batch_size, 128, self.inp_h/16, self.inp_w/16),
			filter_shape=(64, 128, 3, 3),
			zero_pad=True,
			non_linearity=True,
			switch=self.layer7.switch,
			unpoolsize=(2, 2),
			read_file=init,
			W_input=params[13][0],
			b_input=params[13][1]
		)

		# input is 64x16x16
		self.layer15 = deconv_unpool_layer(
			rng,
			input=self.layer14.output,
			image_shape=(batch_size, 64, self.inp_h/8, self.inp_w/8),
			filter_shape=(32, 64, 3, 3),
			zero_pad=True,
			non_linearity=True,
			switch=self.layer6.switch,
			unpoolsize=(2, 2),
			read_file=init,
			W_input=params[14][0],
			b_input=params[14][1]
		)

		# input is 32x32x32
		self.layer16 = deconv_unpool_layer(
			rng,
			input=self.layer15.output,
			image_shape=(batch_size, 32, self.inp_h/4, self.inp_w/4),
			filter_shape=(16, 32, 3, 3),
			zero_pad=True,
			non_linearity=True,
			switch=None,
			unpoolsize=(1, 1),
			read_file=init,
			W_input=params[15][0],
			b_input=params[15][1]
		)

		# input is 16x32x32
		self.layer17 = deconv_unpool_layer(
			rng,
			input=self.layer16.output,
			image_shape=(batch_size, 16, self.inp_h/4, self.inp_w/4),
			filter_shape=(8, 16, 3, 3),
			zero_pad=True,
			non_linearity=True,
			switch=self.layer4.switch,
			unpoolsize=(2, 2),
			read_file=init,
			W_input=params[16][0],
			b_input=params[16][1]
		)

		# input is 8x64x64
		self.layer18 = deconv_unpool_layer(
			rng,
			input=self.layer17.output,
			image_shape=(batch_size, 8, self.inp_h/2, self.inp_w/2),
			filter_shape=(4, 8, 3, 3),
			zero_pad=True,
			non_linearity=True,
			switch=None,
			unpoolsize=(1, 1),
			read_file=init,
			W_input=params[17][0],
			b_input=params[17][1]
		)

		# input is 4x64x64
		self.layer19 = deconv_unpool_layer(
			rng,
			input=self.layer18.output,
			image_shape=(batch_size, 4, self.inp_h/2, self.inp_w/2),
			filter_shape=(2, 4, 3, 3),
			zero_pad=True,
			non_linearity=True,
			switch=self.layer2.switch,
			unpoolsize=(2, 2),
			read_file=init,
			W_input=params[18][0],
			b_input=params[18][1]
		)

		# input is 2x128x128
		self.layer20 = deconv_unpool_layer(
			rng,
			input=self.layer19.output,
			image_shape=(batch_size, 2, self.inp_h, self.inp_w),
			filter_shape=(1, 2, 3, 3),
			zero_pad=True,
			non_linearity=False,
			switch=None,
			unpoolsize=(1, 1),
			read_file=init,
			W_input=params[19][0],
			b_input=params[19][1]
		)
		# output at this level is 1x128x128
		'''
