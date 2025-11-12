
import argparse
import models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

r'''learning rate scheduler types
    - step: Decays the learning rate of each parameter group
            by gamma every step_size epochs.
    - multistep: Decays the learning rate of each parameter group
                 by gamma once the number of epoch reaches one of the milestones.
    - exp: Decays the learning rate of each parameter group by gamma every epoch.
    - cosine: Set the learning rate of each parameter group
              using a cosine annealing schedule.
'''
schedule_types = [
    'step', 'multistep', 'exp', 'cosine'
]

def config(parser):
    r"""configuration settings
    """
    parser = argparse.ArgumentParser(description='AI-Challenge Base Code')
    parser.add_argument('--dataset', metavar='DATA', default='cifar10', help='gtsrb, cifar10, cifar100, imagenet')
    # for model architecture
    parser.add_argument('--model', metavar='ARCH', default='preactresnet')
    parser.add_argument('--layers', default='18', type=str, help="16,16_bn")
    parser.add_argument('--model_all', default='preactresnet18', type=str, help='preactresnet, resnet, vgg, densenet, mobilenet, rexnet')
    parser.add_argument('--width-mult', dest='width_mult', default=1.0, type=float, metavar='WM',
                        help='width multiplier to thin a network '
                             'uniformly at each layer (default: 1.0)')
    parser.add_argument('--depth-mult', default=1.0, type=float, metavar='DM',
                         help='depth multiplier network (rexnet)')
    parser.add_argument('--model-mult', default=0, type=int,
                        help="e.g. efficient type (0 : b0, 1 : b1, 2 : b2 ...)")
    # for dataset
    parser.add_argument('--dataset_path', default='./data', type=str, metavar='PATH',
                        help='where you want to load/save your dataset? (default: ../data)')
    parser.add_argument('--index', type=str)
    parser.add_argument('--ratio', type=float, default=0.3)
    # for model weight
    parser.add_argument('--attack', default=True,  type=bool)
    parser.add_argument('--result_file', default='', type=str, help='')
    # for learning policy

    parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--pin_memory', default=True, type=bool, help='use pin memory?')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate (default: 0.1)',
                        dest='lr')
    parser.add_argument('--warmup-lr', '--warmup-learning-rate', default=0.01, type=float,
                        help='initial learning rate for warmup (default: 0.1)',
                        dest='warmup_lr')
    parser.add_argument('--warmup-lr-epoch', default=50, type=int,
                        help='learning rate warmup period (default: 0)',
                        dest='warmup_lr_epoch')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--wd', '--weight-decay', dest='weight_decay',
                        default=5e-4, type=float,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--nest', '--nesterov', dest='nesterov', action='store_true',
                        help='use nesterov momentum?')
    parser.add_argument('--sched', '--scheduler', dest='scheduler', metavar='TYPE',
                        default='multistep', type=str, choices=schedule_types,
                        help='scheduler: ' +
                             ' | '.join(schedule_types) +
                             ' (default: step)')
    parser.add_argument('--step-size', dest='step_size', default=20,
                        type=int, metavar='STEP',
                        help='period of learning rate decay / '
                             'maximum number of iterations for '
                             'cosine annealing scheduler (default: 30)')
    parser.add_argument('--milestones', metavar='EPOCH', default=[50, 60], type=int, nargs='+',
                        help='list of epoch indices for multi step scheduler '
                             '(must be increasing) (default: 100 150)')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='multiplicative factor of learning rate decay (default: 0.1)')


    # for gpu configuration
    parser.add_argument('-D', '--device', default='cuda', help='use cuda?')
    parser.add_argument('-g', '--gpuids', metavar='GPU', default=[0],
                        type=int, nargs='+',
                        help='GPU IDs for using (default: 0)')
    # specify run type
    parser.add_argument('--run-type', default='evaluate', type=str, metavar='TYPE',
                        help='type of run the main function e.g. train or evaluate (default: train)')
    # for load and save
    parser.add_argument('--save', default='ckpt.pth', type=str, metavar='FILE.pth',
                        help='name of checkpoint for saving model (default: ckpt.pth)')
    #############
    # for pruning
    parser.add_argument('--run-type',default='train', type=str, help='train or evaluate')
    parser.add_argument('--prune', default=True, type=bool, help='Use pruning')
    parser.add_argument('--pruner', default='dcil', type=str,
                        help='method of pruning to apply (default: dcil)')
    parser.add_argument('--prune-freq', dest='prune_freq', default=30, type=int,
                         help='update frequency')
    parser.add_argument('--keep_rate', dest='keep_rate', default=0.9, type=float)
    parser.add_argument('--prune-imp', dest='prune_imp', default='L1', type=str,
                         help='Importance Method : L1, L2, grad, syn')
    parser.add_argument('--initial_cov_rate', type=float, default=0.6)
    parser.add_argument('--final_cov_rate', type=float, default=0.8)
    parser.add_argument('--prun', dest='prune_imp', default='L1', type=str,
                        help='Cuda_num')

    parser.add_argument('--cu_num', default='1', type=str)

    parser.add_argument('--warmup-loss', dest='warmup_loss', default=50, type=int,
                         help='warmup epoch for KD ')
    parser.add_argument('--first_epoch', default=15, type=int,
                        help='init_epoch ')
    parser.add_argument('--target_epoch', default=60, type=int,
                         help='target_training_epoch ')
    parser.add_argument('--print_freq', default=16, type=int)
    parser.add_argument('--txt_name_train', default=16, type=str,
                         help='name ')

    parser.add_argument('--txt_name_test', default=16, type=str,
                         help='name ')


    # evaluation
    parser.add_argument('--print_every', default=10, type=int, metavar='N')
    parser.add_argument('--amp', default=True, type=bool)
    parser.add_argument('--frequency_save', default=1, type=int, metavar='N')
    parser.add_argument('--prefetch', default=False, type=bool)
    parser.add_argument('--non_blocking', default=True, type=bool)
    cfg = parser.parse_args()
    return cfg

