# data/config.py
import argparse


parser = argparse.ArgumentParser(description='LCZ baseline detect 17 examples')

parser.add_argument('--root', default = '/home/devdata/cjs/AI/dataset/', help='dir where data exists')
parser.add_argument('--train_path', default='/home/devdata/cjs/AI/dataset/training.h5', help='training data exists')
parser.add_argument('--val_path', default='/home/devdata/cjs/AI/dataset/validation.h5', help='validation data exists')
parser.add_argument('--test_path', default='/home/devdata/cjs/AI/dataset/round1_test_b_20190104.h5', help='test data exists')

parser.add_argument('--batch_size', type=int, default=256, help='input batch_size for training (default: 128)')
parser.add_argument('--lr_decay', type=float,default=5,help='learning rate decay(default:5)')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate(default:1e-3)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--update_epochs', default=(80, 100, 110), type=list, help='Gamma update epochs for SGD')
parser.add_argument('--epochs', type=int, default=100,help='Number of epochs to train(default)')
parser.add_argument('--alpha', default=1., type=float, help='param for mixup(default:1.)')
parser.add_argument('--mixname', type=str, default='36', help='mixname include two groups(such as: 36 810)')
parser.add_argument('--firstgroup', type=int, default=3, help='mixname include two groups, this stands for first group')
parser.add_argument('--secondgroup', type=int, default=6, help='mixname include two groups, this stands for second group')
parser.add_argument('--kfold', type=int, default=12, help='number of kfold(default:10)')

parser.add_argument('--use_small_model', default=False, type=bool, help='if use small model to trick, set True, else set False')
parser.add_argument('--max_epochs', default=4, type=int, help='Number of training epochs')
parser.add_argument('--gpu', type=str ,default='0', help='gpu id')
parser.add_argument('--resume', default=0, type=int, help='0 means not run from the checkpoint, 1 means run from the checkpoint')
parser.add_argument('--is_onlytest', default=0, type=int, help='0 means train and test, 1 means only test')
parser.add_argument('--model_name', type=str, default='pnasnet5large', help='model name(default: densenet121 )')
parser.add_argument('--stage',type=int, default=0, help='initial stage')
parser.add_argument('--epoch',type=int, default=0, help='initial epoch')
parser.add_argument('--best_precision',type=int, default=0, help='initial best_precision')
parser.add_argument('--lowest_loss',type=int, default=100, help='initial lowest_loss')
parser.add_argument('--print_freq',type=int, default=10, help='every print_freq time starting print')
# parser.add_argument('--stage_epochs', default=(2, 2,2 ,2), type=list, help='stage_epochs')
# parser.add_argument('--stage_epochs', default=(4,4,5,5,4), type=list, help='stage_epochs')
# parser.add_argument('--stage_epochs', default=(5,4,5,5,4), type=list, help='stage_epochs') 
# parser.add_argument('--stage_epochs', default=(5,4,5,5,4), type=list, help='stage_epochs')#1231
# parser.add_argument('--stage_epochs', default=(5,5,5,4,4), type=list, help='stage_epochs')#11
# parser.add_argument('--stage_epochs', default=(7,7,7,5,5), type=list, help='stage_epochs')#11
# parser.add_argument('--stage_epochs', default=(5,4,5,4,5), type=list, help='stage_epochs')#12
# parser.add_argument('--stage_epochs', default=(5,4,5,5,5), type=list, help='stage_epochs')#17
parser.add_argument('--stage_epochs', default=(5,5,5,5,5), type=list, help='stage_epochs')#overfit
# parser.add_argument('--stage_epochs', default=(1,1), type=list, help='stage_epochs')

args = parser.parse_args()
