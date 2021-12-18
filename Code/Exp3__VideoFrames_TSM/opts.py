import argparse

parser = argparse.ArgumentParser(description='PyTorch Smth-Else')
# ========================= Model Configs ==========================
parser.add_argument('--model_name', default='TSM')
parser.add_argument('--size', default=224, type=int, metavar='N',help='primary image input size')
parser.add_argument('--hidden_dim', default=200, type=int, metavar='N',help='hidden neurons for LSTM')
parser.add_argument('--layer_dim', default=2, type=int, metavar='N',help='number of layers for LSTM')
parser.add_argument('--num_classes', default=2, type=int,help='num of class in the model')
parser.add_argument('--freeze_model', default=True, action="store_true", help='Freeze TSM model and train STIN model (default: False)')
parser.add_argument('--unfreeze_layers', default=1, help='Freeze TSM model layers')

# ========================= Learning Configs ==========================
parser.add_argument('--clip_gradient', '-cg', default=10, type=float,
                    metavar='W', help='gradient norm clipping (default: 5)')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start_epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--gamma', default=0.5, type=float, help='decay rate in lr scheduler')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
parser.add_argument('--wd', '--weight-decay', default=10e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')

# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--distributed', action='store_true', help='Run distributed training. Default True')
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU id to use.')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

# ========================= Logging Configs ==========================
parser.add_argument('--tb', default=None, help='Tensorboard logger to use.')
parser.add_argument('--log', default=None, help='File logger to use.')
parser.add_argument('--logdir', default='log_files',
                    help='folder to output tensorboard logs')
parser.add_argument('--logname', default='exp',
                    help='name of the experiment for checkpoints and logs')
parser.add_argument('--ckpt', default='./ckpt',
                    help='folder to output checkpoints')
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--log_freq', '-l', default=10, type=int,
                    metavar='N', help='frequency to write in tensorboard (default: 10)')

parser.add_argument('--dry_run', default=False, action="store_true", help='dry run mode for testing')

parser.add_argument('--pretrained_models_path', default='pretrained_models/', help='path with ckpt to restore')
parser.add_argument('--tsm_resume', default='TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment16_e45.pth', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--temporal_pool', default=False, action="store_true", help='add temporal pooling')
parser.add_argument('--img_feature_dim', default=224, type=int, metavar='N', help='intermediate feature dimension for image-based features')
parser.add_argument('--coord_feature_dim', default=128, type=int, metavar='N', help='intermediate feature dimension for coord-based features')
parser.add_argument('--fine_tune', default=True, help='path with ckpt to restore')
