import numpy as np
import sys, getopt
import os
from glob import glob
import piexif
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import *
from TrackNet import TrackNet
import keras.backend as K
from keras import optimizers
from keras.activations import *
import tensorflow as tf
import cv2
import math

import threading
from threading import Thread
BATCH_SIZE=8
HEIGHT=288
WIDTH=512
mag = 1
sigma = 2.5

class ThreadWithReturnValue(Thread):
	def __init__(self, x_path, y_path, group=None, target=None, name=None,
				args=(), kwargs={}, Verbose=None):
		Thread.__init__(self, group, target, name, args, kwargs)
		self.x_path = x_path
		self.y_path = y_path
	def run(self):
		self.x = np.load(self.x_path)
		self.y = np.load(self.y_path)
	def join(self):
		Thread.join(self)
		return self.x, self.y
#Return the numbers of true positive, true negative, false positive and false negative
def outcome(y_pred, y_true, tol):
	n = y_pred.shape[0]
	i = 0
	TP = TN = FP1 = FP2 = FN = 0
	while i < n:
		for j in range(1):
			if np.amax(y_pred[i][j]) == 0 and np.amax(y_true[i][j]) == 0:
				TN += 1
			elif np.amax(y_pred[i][j]) > 0 and np.amax(y_true[i][j]) == 0:
				FP2 += 1
			elif np.amax(y_pred[i][j]) == 0 and np.amax(y_true[i][j]) > 0:
				FN += 1
			elif np.amax(y_pred[i][j]) > 0 and np.amax(y_true[i][j]) > 0:
				h_pred = y_pred[i][j] * 255
				h_true = y_true[i][j] * 255
				h_pred = h_pred.astype('uint8')
				h_true = h_true.astype('uint8')
				#h_pred
				(cnts, _) = cv2.findContours(h_pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				rects = [cv2.boundingRect(ctr) for ctr in cnts]
				max_area_idx = 0
				max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
				for j in range(len(rects)):
					area = rects[j][2] * rects[j][3]
					if area > max_area:
						max_area_idx = j
						max_area = area
				target = rects[max_area_idx]
				(cx_pred, cy_pred) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))

				#h_true
				(cnts, _) = cv2.findContours(h_true.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				rects = [cv2.boundingRect(ctr) for ctr in cnts]
				max_area_idx = 0
				max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
				for j in range(len(rects)):
					area = rects[j][2] * rects[j][3]
					if area > max_area:
						max_area_idx = j
						max_area = area
				target = rects[max_area_idx]
				(cx_true, cy_true) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))
				dist = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))
				if dist > tol:
					FP1 += 1
				else:
					TP += 1
		i += 1
	return (TP, TN, FP1, FP2, FN)

#Return the values of accuracy, precision and recall
def evaluation(y_pred, y_true, tol):
	(TP, TN, FP1, FP2, FN) = outcome(y_pred, y_true, tol)
	try:
		accuracy = (TP + TN) / (TP + TN + FP1 + FP2 + FN)
	except:
		accuracy = 0
	try:
		precision = TP / (TP + FP1 + FP2)
	except:
		precision = 0
	try:
		recall = TP / (TP + FN)
	except:
		recall = 0
	return (accuracy, precision, recall)


try:
	(opts, args) = getopt.getopt(sys.argv[1:], '', [
		'load_weights=',
		'save_weights=',
		'dataDir=',
		'valDir=',
		'epochs=',
		'tol='
	])
	if len(opts) < 4:
		raise ''
except:
	print('usage: python3 train_TrackNet3.py --load_weights=<previousWeightPath> --save_weights=<newWeightPath> --dataDir=<npyDataDirectory> --epochs=<trainingEpochs> --tol=<toleranceValue>')
	print('argument --load_weights is required only if you want to retrain the model')
	exit(1)

paramCount={
	'load_weights': 0,
	'save_weights': 0,
	'dataDir': 0,
	'valDir': 0,
	'epochs': 0,
	'tol': 0
}

for (opt, arg) in opts:
	if opt == '--load_weights':
		paramCount['load_weights'] += 1
		load_weights = arg
	elif opt == '--save_weights':
		paramCount['save_weights'] += 1
		save_weights = arg
	elif opt == '--dataDir':
		paramCount['dataDir'] += 1
		dataDir = arg
	elif opt == '--valDir':
		paramCount['valDir'] += 1
		valDir = arg
	elif opt == '--epochs':
		paramCount['epochs'] += 1
		epochs = int(arg)
	elif opt == '--tol':
		paramCount['tol'] += 1
		tol = int(arg)
	else:
		print('usage: python3 train_TrackNet3.py --load_weights=<previousWeightPath> --save_weights=<newWeightPath> --dataDir=<npyDataDirectory> --epochs=<trainingEpochs> --tol=<toleranceValue>')
		print('argument --load_weights is required only if you want to retrain the model')
		exit(1)

if paramCount['save_weights'] == 0 or paramCount['dataDir'] == 0 or paramCount['epochs'] == 0 or paramCount['tol'] == 0:
	print('usage: python3 train_TrackNet3.py --load_weights=<previousWeightPath> --save_weights=<newWeightPath> --dataDir=<npyDataDirectory> --epochs=<trainingEpochs> --tol=<toleranceValue>')
	print('argument --load_weights is required only if you want to retrain the model')
	exit(1)

'''
#you can design your own loss function
def custom_loss(y_true, y_pred):
	loss = 
	return loss
'''

#Training for the first time
if paramCount['load_weights'] == 0:
	model=TrackNet(HEIGHT, WIDTH)
	ADADELTA = optimizers.Adadelta(lr=1.0)
	model.compile(loss=['binary_crossentropy'], optimizer=ADADELTA, metrics=['accuracy']) #loss = my_loss ['binary_crossentropy']
#Retraining
else:
	#if you have own loss function 
	#model = load_model(load_weights, custom_objects={'custom_loss':custom_loss})
	model = load_model(load_weights)

r = os.path.abspath(os.path.join(dataDir))
val = os.path.abspath(os.path.join(valDir))
path = glob(os.path.join(r, '*.npy'))
val_path = glob(os.path.join(val, '*.npy'))
num = 53
idx = np.arange(num, dtype='int') + 1
val_num = 53
val_idx = np.arange(val_num, dtype='int') + 1
print('Beginning training......')
skip_data = False
data_loaded = False

for i in range(epochs):
	print('============epoch', i+1, '================')
	np.random.shuffle(idx)
	if (os.path.isfile(os.path.abspath(os.path.join(dataDir, 'x_data_' + str(idx[0]) + '.npy')))):
		path_x = os.path.abspath(os.path.join(dataDir, 'x_data_' + str(idx[0]) + '.npy'))
		path_y = os.path.abspath(os.path.join(dataDir, 'y_data_' + str(idx[0]) + '.npy'))
		# threading
		thread = ThreadWithReturnValue(path_x, path_y)
		thread.start()		
		skip_data = False
	else:
		skip_data = True

	for j, count in zip(idx, range(len(idx))):		
		if not skip_data:    			
			x_train, y_train = thread.join()
			data_loaded = True

		if (count < len(idx) - 1):
			if (not os.path.isfile(os.path.abspath(os.path.join(dataDir, 'x_data_' + str(idx[count+1]) + '.npy')))):
				print('skip ',dataDir, 'x_data_' + str(idx[count+1]) + '.npy')
				skip_data = True
				continue
			skip_data = False
			path_x = os.path.abspath(os.path.join(dataDir, 'x_data_' + str(idx[count+1]) + '.npy'))
			path_y = os.path.abspath(os.path.join(dataDir, 'y_data_' + str(idx[count+1]) + '.npy'))
			thread = ThreadWithReturnValue(path_x, path_y)
			thread.start()
	
		if data_loaded == False:
			continue
		model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1)
		data_loaded = False
		del x_train
		del y_train
	
	#Save intermediate weights during training & Show the outcome of training data so long
	#if (i+1) % 2 == 0:
	if 1:
		model.save(save_weights + '_' + str(i + 1))
		TP = TN = FP1 = FP2 = FN = 0
		for j in val_idx:
			if (not os.path.isfile(os.path.abspath(os.path.join(valDir, 'x_data_' + str(j) + '.npy')))):
				print('skip ',valDir, 'x_data_' + str(j) + '.npy')
				continue
			x_train = np.load(os.path.abspath(os.path.join(valDir, 'x_data_' + str(j) + '.npy')))
			y_train = np.load(os.path.abspath(os.path.join(valDir, 'y_data_' + str(j) + '.npy')))
			y_pred = model.predict(x_train, batch_size=BATCH_SIZE)
			y_pred = y_pred > 0.5
			y_pred = y_pred.astype('float32')
			(tp, tn, fp1, fp2, fn) = outcome(y_pred, y_train, tol)
			TP += tp
			TN += tn
			FP1 += fp1
			FP2 += fp2
			FN += fn
			del x_train
			del y_train
			del y_pred
		print("Outcome of training data of epoch " + str(i+1) + ":")
		print("Number of true positive:", TP)
		print("Number of true negative:", TN)
		print("Number of false positive FP1:", FP1)
		print("Number of false positive FP2:", FP2)
		print("Number of false negative:", FN)
			
print('Saving weights......')
model.save(save_weights)
print('Done......')
