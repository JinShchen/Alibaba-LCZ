import sys
sys.path.append('..')
from data.config import args
import h5py
import numpy as np

def statics_on_band(sen1):

	avr = np.average(sen1, axis=(0,1,2))
	std = np.std(sen1, axis=(0,1,2))
	max = np.max(sen1, axis=(0,1,2))
	min = np.min(sen1, axis=(0,1,2))
	print('avr is: ', avr)
	print('std is: ', std)
	print('max is: ', max)
	print('min is: ', min)
	

if __name__ == '__main__':
	fid = h5py.File(args.val_path, 'r')
	train = np.array(fid['sen1'])
	statics_on_band(train)