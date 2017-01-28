from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt
import timeit

from model import model

import theano
import theano.tensor as T
import cPickle

data_dir = 'yt_cars'
dirs = os.listdir('../'+data_dir+'/')

def read_rgb(file_name, scaled_dims):
	image = Image.open(open(file_name))
	image = image.resize((128, 128), Image.ANTIALIAS)
	image = np.asarray(image, dtype=theano.config.floatX)
	image = image.transpose(2, 0, 1).reshape(1, 3, scaled_dims[0], scaled_dims[1])
	return image	

def read_label(file_name, scaled_dims):
	image = Image.open(open(file_name))
	image = image.resize((128, 128), Image.ANTIALIAS)
	image = np.asarray(image, dtype=theano.config.floatX)
	image = np.sign(image)
	image = image.reshape(1, 1, scaled_dims[0], scaled_dims[1])
	return image

def shared_dataset(data_xy):
	x, y = data_xy
	shared_x = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)
	shared_y = theano.shared(np.asarray(y, dtype=theano.config.floatX), borrow=True)
	return shared_x, shared_y

# generate data with required resolution
img_list = ()
label_list = ()
dimensions = (128, 128)
print '\n'
for i in range(len(dirs)):
	img = read_rgb('../'+data_dir+'/'+dirs[i]+'/rgb.jpg', dimensions)
	label = read_label('../'+data_dir+'/'+dirs[i]+'/labels_128x128.jpg', dimensions)
	img_list+= (img, )
	label_list+= (label, )
	print '\033[F data loaded:', i*100./len(dirs), '%            '
img_list = np.concatenate(img_list, axis=0)
label_list = np.concatenate(label_list, axis=0)

# generate train, valid and test set 
# ratio 3:1:1
train_set = shared_dataset((img_list[0:(3*len(img_list)/5)], label_list[0:(3*len(label_list)/5)]))
valid_set = shared_dataset((img_list[(3*len(img_list)/5):(4*len(img_list)/5)], label_list[3*len(label_list)/5:4*len(label_list)]))
test_set = shared_dataset((img_list[(4*len(img_list)/5):len(img_list)], label_list[(4*len(label_list)/5):len(label_list)]))

# code for training starts here
batch_size = 10 #update

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set
print 'Input shape:', train_x.shape.eval()
print 'Output shape:', train_y.shape.eval()

n_train_batches = train_x.get_value(borrow=True).shape[0]/batch_size
n_valid_batches = valid_x.get_value(borrow=True).shape[0]/batch_size
n_test_batches = test_x.get_value(borrow=True).shape[0]/batch_size

input = T.tensor4('input')
output = T.tensor4('output')
index = T.lscalar()
rng = np.random.RandomState(23455)

load_file = open('trained_params_level_0.pkl', 'r')
params=()
for i in range(8):
	W = theano.shared(np.asarray(cPickle.load(load_file), dtype=theano.config.floatX), borrow=True)
	b = theano.shared(np.asarray(cPickle.load(load_file), dtype=theano.config.floatX), borrow=True)
	params+=([W, b], )
load_file.close()

#dummy_wt = theano.shared(np.asarray(rng.uniform(low=-1., high=-1., size=(1, 1)), dtype=theano.config.floatX), borrow=True)
#params = ([dummy_wt, ]*2, )*8
dnn = model(rng, input, (128, 128), batch_size=batch_size, params=params, init=True)
#dnn_acc = model(rng, input, (128, 128), batch_size=batch_size, params=params, init=True)

learning_rate = 0.01 # update
cost = - T.mean(output*T.log(dnn.layer8.output) + (1-output)*T.log(1-dnn.layer8.output))
#T.mean(T.sqr(dnn.layer20.output-output))
tensor_shape = T.shape(dnn.layer8.output)
#accuracy = T.sum(T.abs_(output-T.round(dnn_acc.layer8.output)))*100./(tensor_shape[0]*tensor_shape[1]*tensor_shape[2]*tensor_shape[3])

validate_model = theano.function([index], cost, givens={
		input: valid_x[index*batch_size: (index+1)*batch_size],
		output: valid_y[index*batch_size: (index+1)*batch_size]
	})

test_model = theano.function([index], cost, givens={
		input: test_x[index*batch_size: (index+1)*batch_size],
		output: test_y[index*batch_size: (index+1)*batch_size]
	})

#test_acc = theano.function([index], accuracy, givens={
#		input: test_x[index*batch_size: (index+1)*batch_size],
#		output: test_y[index*batch_size: (index+1)*batch_size]
#	})

params = dnn.layer1.params + \
			dnn.layer2.params + \
			dnn.layer3.params + \
			dnn.layer4.params + \
			dnn.layer5.params + \
			dnn.layer6.params + \
			dnn.layer7.params + \
			dnn.layer8.params
grads = T.grad(cost, params)
updates = [
	(param_i, param_i - learning_rate*grad_i)
	for param_i, grad_i in zip(params, grads)
]

train_model = theano.function([index], cost, updates=updates, givens={
		input: train_x[index*batch_size: (index+1)*batch_size],
		output: train_y[index*batch_size: (index+1)*batch_size]
	})

patience = 10000
patience_increase = 2
improvement_threshold = 0.995
validation_frequency = min(n_train_batches, patience / 2)
best_validation_loss = np.inf
last_validation_loss = np.inf
best_iter = 0
test_score = 0.
start_time = timeit.default_timer()

epoch = 0
done_looping = False
n_epochs = 95

print 'no. of minibatches:', n_train_batches
try:
	while (epoch < n_epochs) and (not done_looping):
		epoch = epoch+1
		for minibatch_index in range(n_train_batches):
			iter = (epoch-1)*n_train_batches + minibatch_index
			if iter%100 == 0:
				print 'training @iter =', iter 
			cost_ij = train_model(minibatch_index)

			if (iter+1)%validation_frequency == 0:
				validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
				this_validation_loss = np.mean(validation_losses)
				print 'learning_rate: ', learning_rate
				print ('epoch %i, minibatch %i/%i, mean validation error: %f' %(epoch, minibatch_index+1, n_train_batches, this_validation_loss))
	
				'''if this_validation_loss>last_validation_loss:
					learning_rate = learning_rate/2.
					updates = [
						(param_i, param_i - learning_rate*grad_i)
						for param_i, grad_i in zip(params, grads)
					]
					train_model = theano.function([index], cost, updates=updates, givens={
						input: train_x[index*batch_size: (index+1)*batch_size],
						output: train_y[index*batch_size: (index+1)*batch_size]
					})
					epoch = epoch-1
					continue
	
				last_validation_loss = this_validation_loss'''
	
				if this_validation_loss<best_validation_loss:
					if this_validation_loss<best_validation_loss*improvement_threshold:
						patience = max(patience, iter*patience_increase)
	
					best_validation_loss=this_validation_loss
					best_iter=iter
	
				test_losses = [
					test_model(i)
					for i in xrange(n_test_batches)
				]
				test_loss = np.mean(test_losses)
				print '\tepoch %i, minibatch %i/%i, mean test error: %f ' %(epoch, minibatch_index+1, n_train_batches, test_loss)

				#test_accs = [
				#	test_acc(i)
				#	for i in xrange(n_test_batches)
				#]
				#test_accuracy = np.mean(test_accs)
				#print '\tepoch %i, minibatch %i/%i, mean test accuracy(error): %f ' %(epoch, minibatch_index+1, n_train_batches, test_accuracy)
	
				#if patience<=iter:
				#	done_looping=True
				#	break

except KeyboardInterrupt:
	save_file = open('trained_params_level_0_aerial.pkl', 'wb')
	for i in range(len(params)):
		cPickle.dump(params[i].get_value(borrow=True), save_file, -1)
	save_file.close()
	print 'params saved @', save_file
	end_time = timeit.default_timer()
	print 'code ran for %.2f minutes' %((end_time-start_time)/60.)
	exit()
	
end_time = timeit.default_timer()

print 'code ran for %.2f minutes' %((end_time-start_time)/60.)
save_file = open('trained_params_level_0_aerial.pkl', 'wb')
for i in range(len(params)):
	cPickle.dump(params[i].get_value(borrow=True), save_file, -1)
save_file.close()
print 'params saved @', save_file
