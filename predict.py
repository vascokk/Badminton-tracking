import sys
import getopt
import numpy as np
import os
from glob import glob
import piexif
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import *
from TrackNet import TrackNet
import keras.backend as K
from keras import optimizers
from keras.models import *
from keras.layers import *
import cv2
from os.path import isfile, join
from PIL import Image
import time
import math
BATCH_SIZE=1
HEIGHT=288
WIDTH=512
sigma=2.5
mag=1

def EFO_model(input_imgs):

	out = Conv2D(1, (30, 30), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(input_imgs)
	return out

def genHeatMap(w, h, cx, cy, r, mag):
	if cx < 0 or cy < 0:
		return np.zeros((h, w))
	x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
	heatmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
	heatmap[heatmap <= r**2] = 1
	heatmap[heatmap > r**2] = 0
	return heatmap*mag

##############################################################################
#Return the numbers of true positive, true negative, false positive and false negative
def outcome(y_pred, y_true, tol):
	n = y_pred.shape[0]
	i = 0
	TP = TN = FP1 = FP2 = FN = 0
	while i < n:
		h_pred = y_pred[i]*255
		h_true = y_true[i]*255
		h_pred = h_pred.astype('uint8')
		h_true = h_true.astype('uint8')
		if np.amax(h_pred) == 0 and np.amax(h_true) == 0:
			TN += 1
		elif np.amax(h_pred) > 0 and np.amax(h_true) == 0:
			FP2 += 1
		elif np.amax(h_pred) == 0 and np.amax(h_true) > 0:
			FN += 1
		elif np.amax(h_pred) > 0 and np.amax(h_true) > 0:
			#h_pred
			(cnts, _) = cv2.findContours(h_pred[0].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
##############################################################################


#time: in milliseconds
def custom_time(time):
	remain = int(time / 1000)
	ms = (time / 1000) - remain
	s = remain % 60
	s += ms
	remain = int(remain / 60)
	m = remain % 60
	remain = int(remain / 60)
	h = remain
	#Generate custom time string
	cts = ''
	if len(str(h)) >= 2:
		cts += str(h)
	else:
		for i in range(2 - len(str(h))):
			cts += '0'
		cts += str(h)
	
	cts += ':'

	if len(str(m)) >= 2:
		cts += str(m)
	else:
		for i in range(2 - len(str(m))):
			cts += '0'
		cts += str(m)

	cts += ':'

	if len(str(int(s))) == 1:
		cts += '0'
	cts += str(s)

	return cts


try:
	(opts, args) = getopt.getopt(sys.argv[1:], '', [
		'video_name=',
		'load_weights=',
		'label=',
		'tol='
	])
	if len(opts) != 4:
		raise ''
except:
	print('usage: python3 predict.py --video_name=<videoPath> --load_weights=<weightPath> --label=<csvFile> --tol=<toleranceValue>')
	exit(1)

for (opt, arg) in opts:
	if opt == '--video_name':
		videoName = arg
	elif opt == '--load_weights':
		load_weights = arg
	elif opt == '--label':
		labelPath = arg
	elif opt == '--tol':
		tol = arg
	else:
		print('usage: python3 predict.py --video_name=<videoPath> --load_weights=<weightPath> --label=<csvFile> --tol=<toleranceValue>')
		exit(1)
'''
you can design your own loss function
def custom_loss(y_true, y_pred):
	loss = 
	return loss
'''
model = load_model(load_weights) #load_model(load_weights, custom_objects={'custom_loss':custom_loss})

print('Beginning predicting......')

f = open(videoName[:-4]+'_predict.csv', 'w')
f.write('Frame,Visibility,X,Y,Time\n')

cap = cv2.VideoCapture(videoName)
fps = cap.get(cv2.CAP_PROP_FPS)

success, image1 = cap.read()
ratio = image1.shape[0] / HEIGHT

size = (int(WIDTH*ratio), int(HEIGHT*ratio))

if videoName[-3:] == 'avi':
	fourcc = cv2.VideoWriter_fourcc(*'DIVX')
elif videoName[-3:] == 'mp4':
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
else:
	print('usage: video type can only be .avi or .mp4')
	exit(1)

out = cv2.VideoWriter(videoName[:-4]+'_predict'+videoName[-4:], fourcc, fps, size)

out.write(image1)
count = 1
data = pd.read_csv(labelPath)
no = data['Frame'].values
v = data['Visibility'].values
x = data['X'].values
y = data['Y'].values
num = no.shape[0]
##################generate predict video & csv #################################
time_list=[]
TP = TN = FP1 = FP2 = FN = 0
while success:
	if(len(x) == count):
		break
	unit = []
	#Adjust BGR format (cv2) to RGB format (PIL)
	x1 = image1[...,::-1]
	#Convert np arrays to PIL images
	x1 = array_to_img(x1)
	#Resize the images
	x1 = x1.resize(size = (WIDTH, HEIGHT))
	#Convert images to np arrays and adjust to channels first
	x1 = np.moveaxis(img_to_array(x1), -1, 0)

	#Create data
	unit.append(x1[0])
	unit.append(x1[1])
	unit.append(x1[2])
	unit=np.asarray(unit)	
	unit = unit.reshape((1, 3, HEIGHT, WIDTH))
	unit = unit.astype('float32')
	unit /= 255
	start = time.time()
	y_pred = model.predict(unit, batch_size=BATCH_SIZE)
	end = time.time()
	time_list.append(end-start)
	y_pred = y_pred > 0.5
	y_pred = y_pred.astype('float32')
	y_true = []
	if(v[count-1] == 0):
		y_true.append(genHeatMap(WIDTH, HEIGHT, -1, -1, sigma, mag))
	else:
		y_true.append(genHeatMap(WIDTH, HEIGHT, int(x[count-1]/ratio), int(y[count-1]/ratio), sigma, mag))

	(tp, tn, fp1, fp2, fn) = outcome(y_pred, y_true, int(tol))
	TP += tp
	TN += tn
	FP1 += fp1
	FP2 += fp2
	FN += fn

	h_pred = y_pred[0]*255
	h_pred = h_pred.astype('uint8')
	frame_time = custom_time(cap.get(cv2.CAP_PROP_POS_MSEC))
	if np.amax(h_pred) <= 0:
		f.write(str(count)+',0,0,0,'+frame_time+'\n')
		out.write(image1)
	else:
		#h_pred
		(cnts, _) = cv2.findContours(h_pred[0].copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		rects = [cv2.boundingRect(ctr) for ctr in cnts]
		max_area_idx = 0
		max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
		for i in range(len(rects)):
			area = rects[i][2] * rects[i][3]
			if area > max_area:
				max_area_idx = i
				max_area = area
		target = rects[max_area_idx]
		(cx_pred, cy_pred) = (int(ratio*(target[0] + target[2] / 2)), int(ratio*(target[1] + target[3] / 2)))

		f.write(str(count)+',1,'+str(cx_pred)+','+str(cy_pred)+','+frame_time+'\n')
		image1_cp = np.copy(image1)
		cv2.circle(image1_cp, (cx_pred, cy_pred), 5, (0,0,255), -1)
		out.write(image1_cp)
	success, image1 = cap.read()
	count += 1

f.close()
out.release()
total_time = sum(time_list)
################################################################################

################################ACC caculation##################################

print('==========================================================')

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

print("Number of true positive:", TP)
print("Number of true negative:", TN)
print("Number of false positive FP1:", FP1)
print("Number of false positive FP2:", FP2)
print("Number of false negative:", FN)
print("Accuracy:", accuracy)	
print("Precision:", precision)
print("Recall:", recall)
print("Total Time:", total_time)

##################generate predict video & csv #################################

input_imgs = tf.placeholder(shape=[None, 600, 600, 1], dtype=tf.float32)
input_imgs = tf.reshape(input_imgs,[1,1,600,600])
start = time.time()
out = EFO_model(input_imgs)
end = time.time()
EFO_unit = end-start

print('(ACC + Pre + Rec)/3:', (accuracy+precision+recall) / 3)
print('EFO Score:', total_time / EFO_unit)
print('Done......')