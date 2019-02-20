import h5py
from os import path
import numpy as np
from config import args
from torchvision import transforms

TRAIN_OR_VAL = input('train(t) or val(f)? (t for default)')
TRAIN_OR_VAL = (False, True)[TRAIN_OR_VAL != 'f']

def bgr_reader():
    h5file = ('validation.h5', 'training.h5')[TRAIN_OR_VAL]
    with h5py.File(path.join(args.root, h5file), 'r') as f:
        bgrs = np.array(f['sen2'])[:, :, :, :3]
        labels = np.array(f['label'])
    assert bgrs.shape[0] == labels.shape[0], 'num_imgs != num_labels'
    print("bgr bgrs shape is: ",bgrs.shape)
    print('num_imgs: {}'.format(bgrs.shape[0]))
    return bgrs, labels

def writeh5(bgrs, labels):
    h5file = ('validation_bgr.h5', 'training_bgr.h5')[TRAIN_OR_VAL]
    with h5py.File(path.join(args.root, h5file), 'w') as f:
        f.create_dataset('bgr', data=bgrs)
        f.create_dataset('label', data=labels)

def bgrh5_reader():
    h5file = ('validation_bgr.h5', 'training_bgr.h5')[TRAIN_OR_VAL]
    with h5py.File(path.join(args.root, h5file), 'r') as f:
        bgrs = np.array(f['bgr'])
        labels = np.array(f['label'])
    assert bgrs.shape[0] == labels.shape[0], 'num_imgs != num_labels'
    print('shape of bgrs: {}, shape of labels: {}'.format(bgrs.shape, labels.shape))
    print('e.g.: bgr input range: [{}, {}], label: {}'.format(bgrs[0].min(), bgrs[0].max(), labels[0]))

def BGRtoRGB():
    # BGR
    mean = [0.12376287, 0.10928576, 0.10109772]
    std = [0.03959682, 0.04778799, 0.06637737]
    h5file = ('validation_bgr.h5', 'training_bgr.h5')[TRAIN_OR_VAL]

    with h5py.File(path.join(args.root, h5file), 'r') as f:
        bgrs = np.array(f['bgr'])

        labels = np.array(f['label'])
    assert bgrs.shape[0] == labels.shape[0], 'num_imgs != num_labels'
    # data_transform = transforms.Compose([transforms.Normalize(mean = [0.12376287, 0.10928576, 0.10109772], std = [0.03959682, 0.04778799, 0.06637737])])
    h5filenew = ('validation_rgb.h5', 'training_rgb.h5')[TRAIN_OR_VAL]
    with h5py.File(path.join(args.root, h5filenew), 'w') as fin:
        bgrs[:, :, :, [0,2]] = bgrs[:, :, :, [2,0]]
        bgrs = np.transpose(bgrs,[0, 3, 1, 2])
        print("RGB bgrs shape is: ",bgrs.shape)
        print("label shape is: ",labels.shape)
        fin.create_dataset('bgr', data=bgrs)
        fin.create_dataset('label', data=labels)

def splitRGB():
    # BGR
    h5file = ('validation_rgb.h5', 'training_rgb.h5')[TRAIN_OR_VAL]

    with h5py.File(path.join(args.root, h5file), 'r') as f:
        bgrs = np.array(f['bgr'])
        labels = np.array(f['label'])
    assert bgrs.shape[0] == labels.shape[0], 'num_imgs != num_labels'
    # data_transform = transforms.Compose([transforms.Normalize(mean = [0.12376287, 0.10928576, 0.10109772], std = [0.03959682, 0.04778799, 0.06637737])])
    h5filenew = ('validation_rgb_new.h5', 'training_rgb_new.h5')[TRAIN_OR_VAL]
    with h5py.File(path.join(args.root, h5filenew), 'w') as fin:
        bgrs = bgrs[:128, :, :, :]
        labels = labels[:128,:]
        print("RGB bgrs shape is: ",bgrs.shape)
        fin.create_dataset('bgr', data=bgrs)
        fin.create_dataset('label', data=labels)


if __name__ == '__main__':
    #bgrs, labels = bgr_reader()
    #writeh5(bgrs, labels)
    #bgrh5_reader()
    #BGRtoRGB()
    #splitRGB()