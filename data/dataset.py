# data/dataset.py
# -*- coding:utf-8 -*-
from data.config import args
from torch.utils.data import Dataset
import random
import h5py
import cv2
import numpy as np
import torch

class LCZ_TrainData(Dataset):
	def __init__(self, transform=None, target_transform=None):
		super(LCZ_TrainData,self).__init__()
		fid = h5py.File(args.val_path, 'r')
		print("train_path: ",args.val_path)
		bgrs = np.array(fid['sen2'])#[:,:,:,:3]
		#####11
		# sen1 = np.array(fid['sen1'])
		# sen1 = 20 * np.log(np.abs(sen1)+0.001)
		#print('bgrs shape: ', bgrs.shape)
		#print('sen1 shape: ', sen1.shape)
		# bgrs = np.concatenate((bgrs, sen1), axis=3)

		#####
		num = bgrs.shape[0]
		self.rgbs = bgrs[:,:,:,:]#[:,:,:,[2,1,0]]
		_, self.labels = np.where(np.array(fid['label']) == 1)
		print("labels shape: ",self.labels.shape)
		self.transform = transform
		self.target_transform = target_transform

	def __getitem__(self,index):
		rgb = self.rgbs[index]
		if self.transform is not None:
			rgb = self.transform(rgb)
		label = self.labels[index]
		return rgb, label

	def __len__(self):
		return len(self.labels)

	def statics_on_band(self):
		avr = np.average(self.rgbs, axis=(0,1,2))
		std = np.std(self.rgbs, axis=(0,1,2))
		return avr, std

	def statics_on_class(self):
		class_count = np.sum(self.labels, axis=0).tolist() # after one-hot
		return class_count

class LCZ_ValData(Dataset):
	def __init__(self, transform=None, target_transform=None):
		super(LCZ_ValData,self).__init__()
		fid = h5py.File(args.val_path, 'r')
		bgrs = np.array(fid['sen2'])#[:,:,:,:3]
		#####11
		# sen1 = np.array(fid['sen1'])
		# sen1 = 20 * np.log(np.abs(sen1)+0.001)
		# bgrs = np.concatenate((bgrs, sen1), axis=-1)
		#####
		# num = bgrs.shape[0]
		#train_num = int(num*0.9)
		# val_num = num - train_num
		self.rgbs = bgrs[:,:,:,:] # [train_num:,:,:,:]# [:,:,:,[2,1,0]]
		_, self.labels = np.where(np.array(fid['label']) == 1)
		print("labels shape: ",self.labels.shape)
		self.transform = transform
		self.target_transform = target_transform

	def __getitem__(self,index):
		rgb = self.rgbs[index]
		if self.transform is not None:
			rgb =self.transform(rgb)
		label = self.labels[index]
		return rgb, label

	def __len__(self):
		return len(self.labels)

	def statics_on_band(self):
		avr = np.average(self.rgbs, axis=(0,1,2))
		std = np.std(self.rgbs, axis=(0,1,2))
		return avr, std

	def statics_on_class(self):
		class_count = np.sum(self.labels, axis=0).tolist() # after one-hot
		return class_count

class LCZ_TestData(Dataset):
	def __init__(self, transform=None, target_transform=None):
		super(LCZ_TestData,self).__init__()
		fid = h5py.File(args.test_path, 'r')
		bgrs = np.array(fid['sen2'])#[:,:,:,:3]
		#####11
		# sen1 = np.array(fid['sen1'])
		# sen1 = 20 * np.log(np.abs(sen1)+0.001)
		# bgrs = np.concatenate((bgrs, sen1), axis=-1)
		#####
		self.rgbs = bgrs#[:,:,:,[2,1,0]]
		self.transform = transform
		self.target_transform = target_transform

	def __getitem__(self,index):
		rgb = self.rgbs[index]
		if self.transform is not None:
			rgb = self.transform(rgb)
		return rgb,index

	def __len__(self):
		return self.rgbs.shape[0]

	def statics_on_band(self):
		avr = np.average(self.rgbs, axis=(0,1,2))
		std = np.std(self.rgbs, axis=(0,1,2))
		return avr, std


class clockRotate90(object):
	def __init__(self,p=0.5):
		self.p=p
	def __call__(self, rgb):
		if random.random() < 0.5:
			return np.rot90(rgb,-1).deepcopy()
		return rgb

class anticlockRotate90(object):
	def __init__(self,p=0.5):
		self.p=p
	def __call__(self, rgb):
		if random.random() < 0.5:
			return np.rot90(rgb).deepcopy()
		return rgb

class Resize(object):
	def __init__(self,size):
		if type(size) is tuple:
			self.size = size
		else:
			self.size = (size,size)
	def __call__(self, rgb):
		rgb = cv2.resize(rgb, self.size, interpolation=cv2.INTER_CUBIC)
		return rgb

class CenterCrop(object):
	def __init__(self, size):
		if type(size) is tuple:
			self.size = size
		else:
			self.size = (size,size)
	def __call__(self, rgb):
		h, w, c = rgb.reshape
		th, tw = self.size
		i = int(round((h-th)/2.))
		j = int(round((w-tw)/2.))
		return rgb[i:i+th, j:j+tw,:]
class RandomCrop(object):
	def __init__(self, size, padding=0, pad_if_needed=False):
		if padding != 0 or pad_if_needed:
			raise NotImplementedError()
		if type(size) is tuple:
			self.size = size
		else:
			self.size = (size, size)
		self.padding = padding
		self.pad_if_needed = pad_if_needed

	@staticmethod
	def get_params(rgb, outpust_size):
		h, w, w = rgb.shape
		th, tw = outpust_size
		assert h >= th and w >= tw
		if w == tw and h == th:
			return 0, 0, h, w
		i = random.randint(0, h-th)
		j = random.randint(0, w-tw)
		return i, j, th, tw

	def __call__(self, rgb):
		i, j, h, w = self.get_params(rgb, self.size)
		return rgb[i:i+h,j:j+w]

	def __repr__(self):
		return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size,self.padding)

class RandomHorizontalFlip(object):
	def __init__(self, p=0.5):
		self.p = p
	def __call__(self,rgb):
		if random.random() < self.p:
			return np.flip(rgb,1) # 1.水平，0.垂直
		return rgb
	def __repr__(self):
		return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomVerticalFlip(object):
	def __init__(self, p=0.5):
		self.p = p
	def __call__(self,rgb):
		if random.random() < self.p:
			return np.flip(rgb,0) # 1.水平，0.垂直
		return rgb
	def __repr__(self):
		return self.__class__.__name__ + '(p={})'.format(self.p)

class ToTensor(object): ## rgb shape??
	def __init__(self):
		pass
	def __call__(self, rgb):
		
		rgb = torch.from_numpy(rgb.transpose((2,1,0)).astype(np.float32))
		return rgb

class Normalize(object):
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std
	def __call__(self, rgb):
		for t, m ,s in zip(rgb, self.mean, self.std):
			t.sub_(m).div_(s)
		return rgb



