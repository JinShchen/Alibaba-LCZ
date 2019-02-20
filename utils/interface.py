
# utils/interface.py
#-*- coding:utf-8 -*-
from data.config import args
from collections import OrderedDict
from tqdm import tqdm
import torch.nn as nn
import time
import torch
import datetime
import numpy as np
import os
from resnet34.net.chenNet import chenNet
from sklearn.preprocessing import LabelEncoder


class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val*n
		self.count += n
		self.avg = self.sum / self.count


# train function
def train(train_loader, model, criterion, optimizer, epoch):
	print("starting train...")
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	acc = AverageMeter()

	model.train()

	end = time.time()

	for i, (images, target) in enumerate(train_loader):
		# measure data loading
		data_time.update(time.time() - end)
		image_var = torch.tensor(images).cuda(async=True)
		label = torch.tensor(target).cuda(async=True)

		# compute y_pred
		y_pred = model(image_var) #y_pred.data??
		loss = criterion(y_pred,label)

		# measure accuracy and record loss
		prec, PRE_COUNT, _ = accuracy(y_pred.data, target, topk=(1,1))
		losses.update(loss.item(), images.size(0)) #loss.item()? images.size(0)?
		acc.update(prec, PRE_COUNT)

		# compute gradient and do SGD step
		optimizer.zero_grad() # initialize with zero gradient
		loss.backward()
		optimizer.step()

		# measure eclapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				'Accuray {acc.val:.3f} ({acc.avg:.3f})'.format(epoch, i, len(train_loader), batch_time=batch_time,data_time=data_time,loss=losses, acc=acc))
	return acc.avg, losses.avg



def labeltostr(label):
	str = ''
	assert(label<17)
	for i in range(17):
		if i != 16:
			if i != label:
				str += '0,'
			else:
				str += '1,'
		else:
			if i != label:
				str += '0\n'
			else:
				str += '1\n'
	return str

def accuracy(y_pred, y_actual, topk=(1,)): # topk??
	final_acc = 0
	maxk = max(topk)

	PRED_COUNT = y_actual.size(0) # ?
	PRED_CORRECT_COUNT = 0
	prob, pred = y_pred.topk(maxk, 1, True, True) # 调用了tensor里的topk方法：第一个输入为计算tOP MAXK的结果；第二个输入时dim，1表示按行；第三个True表示largest=True；第四个True表示sorted

	for j in range(pred.size(0)):
		if int(y_actual[j]) == int(pred[j]):
			PRED_CORRECT_COUNT += 1
		if PRED_COUNT == 0:
			final_acc = 0
		else:
			final_acc = PRED_CORRECT_COUNT / PRED_COUNT
	return final_acc * 100, PRED_COUNT, pred

# test function
def test(test_loader, model, times_t):

	model.eval()
	dict = {}
	label_tmp = 0
	images_tmp = 0
	times = 0
	for i, (images, order) in enumerate(tqdm(test_loader)): # order??
		image_var = torch.tensor(images, requires_grad=False)

		with torch.no_grad():
			y_pred = model(image_var)
			
			smax = nn.Softmax(1) #dim
			
			smax_out = smax(y_pred)
		label = np.argmax(smax_out.cpu().numpy(),axis=1)

		order = order.cpu().numpy()
		for i in range(0, label.shape[0]):
			x = order[i]
			y = label[i]
			dict[order[i]] = label[i]

		if times == 0:
			label_tmp = label
			images_tmp = images
		else:
			label_tmp = np.concatenate((label_tmp, label), axis=0)
			images_tmp = np.vstack((images_tmp, images))
		times += 1

		
	submission = "submit/submit_" +str(times_t) + '_' + args.model_name + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"

	with open(submission, 'w') as f:
		for i in range(0, len(dict)):
			label = dict[i]
			str1 = labeltostr(label)
			f.write(str1)
	return label_tmp
