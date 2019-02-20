import numpy as np
import torch
import h5py
from collections import OrderedDict


# read val
def resample(train_loader, val_loaders):
	total_num = val_loader.labels.shape[0]
	new_val_num = int(np.ceil(total_num*0.4))
	train_loader.rgbs = np.vstack((train_loader.rgbs, val_loader.rgbs[:new_val_num,:,:,:]))
	train_loader.labels = np.concatenate((train_loader.labels, val_loader.labels[:new_val_num]), axis=0)
	val_loader.rgbs = val_loader.rgbs[new_val_num:,:,:]
	val_loader.labels = val_loader.labels[new_val_num:]
	return train_loader,val_loader

'''
# read train
def resample(train_loader, val_loader):
	total_num = train_loader.labels.shape[0]
	new_val_num = int(np.ceil(total_num*0.2))
	train_loader.rgbs = np.vstack((train_loader.rgbs[new_val_num:,:,:,:], val_loader.rgbs))
	train_loader.labels = np.concatenate((train_loader.labels[new_val_num:], val_loader.labels), axis=0)
	val_loader.rgbs = train_loader.rgbs[:new_val_num,:,:,:]
	val_loader.labels = train_loader.labels[:new_val_num]
	return train_loader,val_loader
'''
def make_weights_for_balanced_classes(label, nclasses):
	print("starting making weights for balanced classes. ")
	count = [0] * nclasses
	for item in label:
		count[item] += 1
	weight_per_class = [0.] * nclasses
	N = float(sum(count))
	for i in range(nclasses):
		weight_per_class[i] = N/float(count[i])
	weight = [0] * label.shape[0]
	for idx, val in enumerate(label):
		weight[idx] = weight_per_class[val]
	return weight


def findmax(label):
	print("starting finding max")
	dic = OrderedDict()
	for i in range(0, label.shape[0]):
		dic[label[i]] += 1
	return dic

def weighted_data_generate(data, label):
	np.random.seed(666)
	print('data shape is: ',data.shape)
	print('label shape is: ',label.shape)
	
	# all_data = np.concatenate((data, label), axis=1)
	label_sum = findmax(label)
	max_label = max(label_sum.values())
	# label_sum = np.sum(label,axis = 0)
	# max_label = np.max(label_sum)
	print("label_sum: ", label_sum)
	print("max_label: ", max_label)
	for i in range(0, data.shape[0]):
		now_less_count = max_label - label_sum[i]
		bingo = np.random.choice(a=label_sum, size=now_less_count, replace=True)
		for j in range(0, now_less_count):
			data = np.vstack((data, data[bingo]))
			label = np.vstack((label, label[bingo]))

	print('new_data shape is: ',data)
	print('new_label shape is: ',label)
	return data, label

if __name__ == '__main__':
	val = '/home/devdata/cjs/AI/dataset/training.h5'
	train = '/home/devdata/cjs/AI/dataset/training.h5'
	fid = h5py.File(train, 'r')
	weighted_data(np.array(fid['label']))

	bgrs = np.array(fid['sen2'])[:,:,:,:3]
	# train, val = resample(bgrs, bgrs)
