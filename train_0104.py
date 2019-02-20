# train.py
# -*- coding:utf-8 -*-
# /home/cjs/anaconda3/envs/cuda_test/lib/python3.6/site-packages/pretrainedmodels/models/pnas
# /home/cjs/anaconda3/envs/cuda_test/lib/python3.6/site-packages/torchvision/models/resnet.py
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from cnn_finetune import make_model
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import random
import shutil
import time
import os
from utils.resample import *
from utils.focal_loss_pytorch.focalloss import *
from sklearn.metrics import confusion_matrix

from data.config import args
from utils.interface import *
from data.dataset import *
from sklearn.model_selection import StratifiedKFold

def main(args):
	np.random.seed(666)
	torch.manual_seed(666)
	torch.cuda.manual_seed_all(666)
	random.seed(666)

	# dir
	file_name = os.path.basename(__file__).split('.')[0] #train
	if not os.path.exists('./model/%s' % file_name):
		os.makedirs('./model/%s' % file_name)
	if not os.path.exists('./result/%s' % file_name):
		os.makedirs('./result/%s' % file_name)

	# log
	if not os.path.exists('./result/%s.txt' % file_name):
		with open('./result/%s.txt' % file_name, 'w') as acc_file:
			pass
	with open('./result/%s.txt' % file_name, 'a') as acc_file:
		acc_file.write('\n%s %s\n' % (time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())),file_name))

	def save_checkpoint(state, is_best, is_lowest_loss, times):
		print("save...")
		filename='./model/%s/%s/checkpoint.pth.tar' % (file_name, times)
		torch.save(state, filename)
		if is_best:
			shutil.copyfile(filename, './model/%s/%s/model_best.pth.tar' % (file_name, times))
		if is_lowest_loss:
			shutil.copyfile(filename, './model/%s/%s/lowest_loss.pth.tar' % (file_name, times))

	def adjust_learning_rate_sgd():
		nonlocal lr
		lr = lr / lr_decay
		return optim.SGD(model.parameters(), lr, momentum=args.momentum)

	def adjust_learning_rate():
		nonlocal lr
		lr = lr / lr_decay
		return optim.Adam(model.parameters(), lr, weight_decay=weight_decay, amsgrad=True)


	# validation function
	def validation(val_loader, model, criterion):
		print("starting validation...")
		batch_time = AverageMeter()
		losses = AverageMeter()
		acc = AverageMeter()

		model.eval()

		end = time.time()
		for i, (images, labels) in enumerate(val_loader):
			# measure data loading
			image_var = torch.tensor(images).cuda(async=True)
			target = torch.tensor(labels).cuda(async=True)

			with torch.no_grad():
				y_pred = model(image_var)
				loss = criterion(y_pred, target)

			# measure accuracy and record loss
			prec, PRE_COUNT, pre_label = accuracy(y_pred.cpu().data, labels, topk=(1,1))
			if i==0:
				fisrt = labels
				second = pre_label.flatten()
			else:
				fisrt = np.concatenate((fisrt, labels), axis=0)
				second = np.concatenate((second, pre_label.flatten()), axis=0)

			losses.update(loss.item(), images.size(0)) #loss.item()? images.size(0)?
			acc.update(prec, PRE_COUNT)

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				print('Validation: [{0}/{1}]\t'
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					'Accuray {acc.val:.3f} ({acc.avg:.3f})'.format(i, len(val_loader), batch_time=batch_time, loss=losses, acc=acc))

		
		# confusion
		C = confusion_matrix(fisrt, second)
		

		# print(C)
		nowtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
		with open('./CM/' + nowtime + 'CM.txt', 'a') as fcm:
			for i in range(C.shape[0]):
				fcm.write(str(C[i]))
				fcm.write('\n')
			fcm.write('\n\n')
		print(' * Accuray {acc.avg:.3f}'.format(acc=acc), '(Previous Best Acc: %.3f)' % best_precision,
			' * Loss {loss.avg:.3f}'.format(loss=losses), '(Previous Lowest Loss: %.3f)' % lowest_loss)
		return acc.avg, losses.avg


		# arg
	lr = args.lr
	lr_decay = args.lr_decay
	weight_decay = args.weight_decay # 正则化
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	batch_size = args.batch_size
	workers = batch_size // 2
	# epoch数量，分stage跑
	stage_epochs = args.stage_epochs

	# initial
	stage = args.stage
	start_epoch = args.epoch
	total_epochs = sum(stage_epochs)
	best_precision = args.best_precision
	lowest_loss = args.lowest_loss
	print_freq = args.print_freq
	evaluate = False
	resume = args.resume
	is_onlytest = args.is_onlytest

	# model
	model = make_model(args.model_name, pretrained=False, num_classes=17, dropout_p=0.5)
	model = nn.DataParallel(model).cuda()

	optimizer = optim.Adam(model.parameters(), lr, weight_decay=weight_decay,amsgrad=True)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=total_epochs, eta_min=0, last_epoch=-1)

	if resume:
		checkpoint_path = './model/%s/checkpoint.pth.tar' % file_name
		if os.path.isfile(checkpoint_path):
			print("=> loading checkpoint '{}'".format(checkpoint_path))
			checkpoint = torch.load(checkpoint_path)
			start_epoch = checkpoint['epoch'] + 1
			best_precision = checkpoint['best_precision']
			lowest_loss = checkpoint['lowest_loss']
			stage = checkpoint['stage']
			lr = checkpoint['lr']
			model.load_state_dict(checkpoint['state_dict'])
			# 如果终端点恰好为转换stage的点，需要特殊处理
			if start_epoch in np.cumsum(stage_epochs)[:-1]:
				stage += 1
				scheduler.step()
				# optimizer = adjust_learning_rate()
				model.load_state_dict(torch.load('./model/%s/model_best.pth.tar' % file_name)['state_dict'])
			print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(resume))

	# ImageNet
	# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
	ori_train_data = LCZ_TrainData(transform = transforms.Compose([
								RandomHorizontalFlip(),
								RandomVerticalFlip(),
								#Resize((331, 331)),
								ToTensor(),
								Normalize(mean=[0.129, 0.116, 0.112, 0.123, 0.164, 0.185, 0.178, 0.200, 0.173, 0.127], 
									std=[0.038, 0.043, 0.058, 0.054, 0.064, 0.076, 0.078, 0.085, 0.094, 0.837])]))
								#Normalize(mean=[0.101, 0.109, 0.124], std=[0.066, 0.048, 0.040])])) 
	val_data = LCZ_ValData(transform = transforms.Compose([
								#Resize((331, 331)),
								ToTensor(),
								Normalize(mean=[0.129, 0.116, 0.112, 0.123, 0.164, 0.185, 0.178, 0.200, 0.173, 0.127], 
									std=[0.038, 0.043, 0.058, 0.054, 0.064, 0.076, 0.078, 0.085, 0.094, 0.837])]))
	test_data = LCZ_TestData(transform = transforms.Compose([
								#Resize((331, 331)),
								ToTensor(),
								Normalize(mean=[0.129, 0.116, 0.112, 0.123, 0.164, 0.185, 0.178, 0.200, 0.173, 0.127], 
									std=[0.038, 0.043, 0.058, 0.054, 0.064, 0.076, 0.078, 0.085, 0.094, 0.837])]))

	train_data = LCZ_TrainData(transform = transforms.Compose([
								RandomHorizontalFlip(),
								RandomVerticalFlip(),
								#Resize((331, 331)),
								ToTensor(),
								Normalize(mean=[0.129, 0.116, 0.112, 0.123, 0.164, 0.185, 0.178, 0.200, 0.173, 0.127], 
									std=[0.038, 0.043, 0.058, 0.054, 0.064, 0.076, 0.078, 0.085, 0.094, 0.837])]))

	train_data, val_data = resample(train_data, val_data)
	# train_data.rgbs, train_data.labels = weighted_data_generate(train_data.rgbs, train_data.labels)

	sfold = StratifiedKFold(n_splits=args.kfold, random_state=666, shuffle=False)
	times = 0 # mark wich folds number
	for train_index, val_index in sfold.split(ori_train_data.rgbs, ori_train_data.labels):
		# initial
		stage = args.stage
		start_epoch = args.epoch
		total_epochs = sum(stage_epochs)
		best_precision = args.best_precision
		lowest_loss = args.lowest_loss
		print_freq = args.print_freq
		evaluate = False
		resume = args.resume
		is_onlytest = args.is_onlytest

		# model
		model = make_model(args.model_name, pretrained=False, num_classes=17, dropout_p=0.5)
		model = nn.DataParallel(model).cuda()

		optimizer = optim.Adam(model.parameters(), lr, weight_decay=weight_decay,amsgrad=True)
		scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=total_epochs, eta_min=0, last_epoch=-1)

		train_data.rgbs = ori_train_data.rgbs[train_index]
		train_data.labels = ori_train_data.labels[train_index]
		val_data.rgbs = ori_train_data.rgbs[val_index]
		val_data.labels = ori_train_data.labels[val_index]

		train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=workers)
		val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=workers)
		test_loader = DataLoader(test_data, batch_size=batch_size*2, shuffle=False, pin_memory=False, num_workers=workers)
        
		# cross entropy
		criterion = FocalLoss().cuda()
		# criterion = nn.CrossEntropyLoss().cuda()


		writer = SummaryWriter()

		if is_onlytest:
			dict = {}
			group_label = 0
			for i in range(0, args.kfold):
				best_model = torch.load('./model/%s/%s/model_best.pth.tar' % (file_name, str(i)))
				model.load_state_dict(best_model['state_dict'])
				group_label_tmp = test(test_loader=test_loader, model=model, times_t=i)
				if i == 0:
					group_label = group_label_tmp
					print('group_label shape is: ', group_label.shape)
				else:
					group_label = np.vstack((group_label, group_label_tmp))
					print('group_label shape is: ', group_label.shape)
			group_label = group_label.T
			print('after group_label shape is: ', group_label.shape)
			for i in range(0, group_label.shape[0]):
				line = np.argmax(np.bincount(group_label[i]))
				dict[i] = line

			submission = "submit/all_submit_" + args.model_name + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"

			with open(submission, 'w') as f:
				for i in range(0, len(dict)):
					label = dict[i]
					str1 = labeltostr(label)
					f.write(str1)

			# 释放GPU缓存
			torch.cuda.empty_cache()
			return
		else:
			pass

		if evaluate:
			validation(val_loader=val_loader, model=model, criterion=criterion)
		else:
			# train 开始训练
			with open('./result/%s.txt' % file_name, 'a') as acc_file:
				acc_file.write('----------------------------------')
				acc_file.write('lr: %s\n kfold: %2d\n batch_size:%3d\n' % (args.lr, args.kfold, args.batch_size))
				acc_file.write(str(args.stage_epochs))
			for epoch in range(start_epoch, total_epochs):
				scheduler.step()
				train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch)
				precision, avg_loss = validation(val_loader, model, criterion)
				writer.add_scalars('loss/scalar_group', {'train_loss':train_loss, 'val_loss':avg_loss}, epoch)
				writer.add_scalars('accuracy/scalar_group', {'train_acc':train_acc, 'val_acc':precision}, epoch)
				writer.add_scalars('loss_accuracy/scalar_group', {'train_loss':train_loss, 'val_loss':precision}, epoch)

				# 在日志文件中记录每个epoch的精度和loss
				with open('./result/%s.txt' % file_name, 'a') as acc_file:
					acc_file.write('Times: %2d, Epoch: %2d, Prcision: %.8f, Loss: %.8f, T_Prcision: %.8f, T_Loss: %.8f\n' % (times, epoch, precision, avg_loss, train_acc, train_loss))

				is_best = precision > best_precision
				is_lowest_loss = avg_loss < lowest_loss
				best_precision = max(precision, best_precision)
				lowest_loss = min(avg_loss, lowest_loss)
				state = {
					'epoch': epoch,
					'state_dict': model.state_dict(),
					'best_precision': best_precision,
					'lowest_loss': lowest_loss,
					'stage': stage,
					'lr': lr,
				}
				save_checkpoint(state, is_best, is_lowest_loss, str(times))

				# 判断是否进行下一个stage
				if (epoch + 1) in np.cumsum(stage_epochs)[:-1]:
					stage +=1
					# optimizer = adjust_learning_rate()
					model.load_state_dict(torch.load('./model/%s/%s/model_best.pth.tar' % (file_name, str(times)))['state_dict'])
					print('Step into next stage')

					with open('./results/%s.txt' % file_name, 'a') as acc_file:
						acc_file.write('------------------Step into next stage-------------------------\n')
			writer.export_scalars_to_json("./all_scalars.json")
			writer.close()

			with open('./result/%s.txt' % file_name, 'a') as acc_file:
				acc_file.write('* best acc: %.8f %s\n' % (best_precision, os.path.basename(__file__)))

			with open('./result/best_acc.txt', 'a') as acc_file:
				acc_file.write('%s * best acc: %.8f %s\n' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), best_precision,os.path.basename(__file__)))

		times += 1
		
	dict = {}
	group_label = 0
	for i in range(0, args.kfold):
		best_model = torch.load('./model/%s/%s/model_best.pth.tar' % (file_name, str(i)))
		model.load_state_dict(best_model['state_dict'])
		group_label_tmp = test(test_loader=test_loader, model=model, times_t=i)
		if i == 0:
			group_label = group_label_tmp
		else:
			group_label = np.vstack((group_label, group_label_tmp))
	group_label = group_label.T
	print('group_label shape is: ', group_label.shape)
	for i in range(0, group_label.shape[0]):
		line = np.argmax(np.bincount(group_label[i]))
		dict[i] = line

	submission = "submit/all_submit_" + args.model_name + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"

	with open(submission, 'w') as f:
		for i in range(0, len(dict)):
			label = dict[i]
			str1 = labeltostr(label)
			f.write(str1)

	torch.cuda.empty_cache()

if __name__ == '__main__':
	main(args)
