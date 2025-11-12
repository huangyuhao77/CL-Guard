import argparse
import copy
import logging
import os
import sys
import time
from pprint import pformat
import random

sys.path.append('../')
sys.path.append(os.getcwd())

import numpy as np
import torch
import yaml
from torch import nn, optim

from torch.utils.data import RandomSampler, DataLoader

from utils.dataset_and_transform_generate import get_num_classes, get_input_shape, get_transform, \
    dataset_and_transform_generate
from utils.fix_random import fix_random
from utils.model_trainer_generate import generate_cls_model
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform
from utils.choose_index import choose_index
from utils.log_assist import get_git_info
from utils.save_load_attack import load_attack_result

import lrp, PA
from lrp import trace
from lrp.patterns import fit_patternnet_positive
from lrp.functional.utils import store_patterns, load_patterns
from PA import MaskConv2d, load_state_dict
from run_prune_utils import AverageMeter, ProgressMeter, accuracy, set_scheduler, save_model, save_summary, \
    load_state_dict_v2, get_transform_extends
from utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer, ModelTrainerCLS_v2
import torch.nn.functional as F
from torch.autograd import Variable


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T = 2
        predict = F.log_softmax(pred / T, dim=1)
        target_data = F.softmax(label / T, dim=1)
        target_data = target_data + 10 ** (-7)
        target = Variable(target_data.data.to(args.device), requires_grad=False)
        loss = T * T * ((target * (target.log() - predict)).sum(1).sum() / target.size()[0])
        return loss


criterion_kl = KLLoss().cuda()
schedule_types = [
    'step', 'multistep', 'exp', 'cosine'
]

def all_acc(preds:torch.Tensor,
        labels:torch.Tensor,):
    if len(preds) == 0 or len(labels) == 0:
        logging.warning("zero len array in func all_acc(), return None!")
        return None
    return preds.eq(labels).sum().item() / len(preds)

def add_gaussian_noise(args, tensor, mean=0, std=1.0):
    noise = torch.randn(tensor.size(), device=args.device) * std + mean
    noisy_tensor = tensor + noise
    return noisy_tensor

def set_PureCleantrainer(model):
    trainer = PureCleanModelTrainer(
        model,
    )
    return trainer

class LRP_path_dete():

    def __init__(self, args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)

        defaults.update({k: v for k, v in args.__dict__.items() if v is not None})

        args.__dict__ = defaults

        args.terminal_info = sys.argv

        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        self.args = args

        if 'result_file' in args.__dict__:
            if args.result_file is not None:
                self.set_result(args.result_file)

    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument('--checkpoint_save', type=str, help='the location of checkpoint where model is saved')
        parser.add_argument('--pin_memory', type=bool, help='whether to use pin memory')
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--dataset_path", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny')
        parser.add_argument('--result_file', type=str, help='the location of result')

        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)

        parser.add_argument('--model', type=str, help='resnet18')

        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--yaml_path', type=str, default="../config/clguard/default.yaml",
                            help='the path of yaml')

        parser.add_argument('--acc_ratio', type=float, help='the tolerance ration of the clean accuracy')
        parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')

        parser.add_argument('--index', type=str, help='index of clean data')
        parser.add_argument('--attack', type=bool, help='whether attack')
        parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel')
        parser.add_argument('--runtype', type=str, help='train or evaluate')
        parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                            metavar='LR', help='initial learning rate (default: 0.1)',
                            dest='lr')
        parser.add_argument('--warmup-lr', '--warmup-learning-rate', default=0.01, type=float,
                            help='initial learning rate for warmup (default: 0.1)',
                            dest='warmup_lr')
        parser.add_argument('--warmup_lr_epoch', type=int,
                            help='learning rate warmup period (default: 0)')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum (default: 0.9)')
        parser.add_argument('--wd', '--weight-decay', dest='weight_decay',
                            default=5e-4, type=float,
                            help='weight decay (default: 5e-4)')
        parser.add_argument('--nesterov', action='store_true', help='use nesterov momentum?')
        parser.add_argument('--scheduler', metavar='TYPE',
                            default='multistep', type=str, choices=schedule_types,
                            help='scheduler: ' +
                                 ' | '.join(schedule_types) +
                                 ' (default: step)')
        parser.add_argument('--step-size', dest='step_size',
                            type=int, metavar='STEP',
                            help='period of learning rate decay / '
                                 'maximum number of iterations for '
                                 'cosine annealing scheduler (default: 30)')
        parser.add_argument('--milestones', metavar='EPOCH', type=int, nargs='+',
                            help='list of epoch indices for multi step scheduler '
                                 '(must be increasing) (default: 100 150)')
        parser.add_argument('--gamma', default=0.1, type=float,
                            help='multiplicative factor of learning rate decay (default: 0.1)')

        parser.add_argument('--pruner', default='dcil', type=str,
                            help='method of pruning to apply (default: dcil)')
        parser.add_argument('--prune_freq', type=int,
                            help='update frequency')
        parser.add_argument('--prune_imp', type=str,
                            help='Importance Method : L1, L2, grad, syn')
        parser.add_argument('--initial_cov_rate', type=float)
        parser.add_argument('--final_cov_rate', type=float)
        parser.add_argument('--warmup_loss', default=50, type=int,
                            help='warmup epoch for KD ')
        parser.add_argument('--first_epoch', type=int,
                            help='init_epoch ')
        parser.add_argument('--target_epoch', type=int,
                            help='target_training_epoch ')
        parser.add_argument('--print_freq', type=int)
        parser.add_argument('--txt_name_train', type=str,
                            help='name ')

        parser.add_argument('--txt_name_test', type=str,
                            help='name ')

    def set_result(self, result_file):
        attack_file = '../record/' + result_file
        save_path = '../record/' + result_file + '/defense/lrp_path/'
        save_file = '../record/' + result_file + f'/defense/epochs_{args.target_epoch}_sparcity_{args.final_cov_rate}_initial_cov_rate/'
        npy_save_file = '../record/' + result_file + '/defense/lrp_path/checkpoint/'

        if not (os.path.exists(save_file)):
            os.makedirs(save_file)
        if not (os.path.exists(save_path)):
            os.makedirs(save_path)
        self.args.save_path = save_path
        self.args.save_file = save_file
        self.args.npy_save_file = npy_save_file
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + '/checkpoint/'
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save)
        if self.args.log is None:
            self.args.log = save_file + 'defense/log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)

        if args.attack == False:
            self.result = torch.load(attack_file + "/clean_model.pth")
        else:
            self.result = load_attack_result(attack_file + '/attack_result.pt')

    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()
        fileHandler = logging.FileHandler(
            args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))

        try:
            logging.info(pformat(get_git_info()))
        except:
            logging.info('Getting git info fails.')

    def train_first(self, args, train_loader, epoch, model, criterion, optimizer, scheduler, **kwargs):
        r"""Train model each epoch
        """
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(len(train_loader), batch_time, data_time,
                                 losses, top1, top5, prefix="Epoch: [{}]".format(epoch))

        model.train()

        end = time.time()
        # loader_len = len(train_loader)

        for i, (input, target, *other_info) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if args.device == 'cuda':
                input = input.to(args.device, non_blocking=True)
                target = target.to(args.device, non_blocking=True)

            output = model(input, 0, 0)
            output_full = model(input, 4, 0)

            loss = criterion(output, target)
            # criterion(output_full, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            progress.print(i)
            end = time.time()
            # end of one mini-batch
        print('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        logging.info('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

        return top1.avg, top5.avg

    def train_second(self, args, train_loader, epoch, model, criterion, optimizer, scheduler, **kwargs):
        r"""Train model each epoch
        """
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(len(train_loader), batch_time, data_time,
                                 losses, top1, top5, prefix="Epoch: [{}]".format(epoch))

        model.train()

        end = time.time()
        loader_len = len(train_loader)

        for i, (input, target, *other_info) in enumerate(train_loader):
            scheduler.step(globals()['iterations'] / loader_len)
            # measure data loading time
            data_time.update(time.time() - end)

            if args.device == 'cuda':
                input = input.to(args.device, non_blocking=True)
                target = target.to(args.device, non_blocking=True)

            change_ratio = (args.final_cov_rate - args.initial_cov_rate) / (
                    args.target_epoch - 10 - args.first_epoch)
            current_cov_rate = (epoch - args.first_epoch) * change_ratio + args.initial_cov_rate
            if epoch > args.target_epoch - 10:
                current_cov_rate = args.final_cov_rate

            if globals()['iterations'] % args.prune_freq == 0:
                threshold = PA.get_weight_threshold(model, current_cov_rate, args)
                PA.weight_prune(model, threshold, args)

            output = model(input, 0, 1)
            output_full = model(input, 1, 1)

            if args.mode_type == 'Ablation2':
                loss = criterion(output, target)
            else:
                if epoch < args.warmup_loss:
                    loss = criterion(output, target) + criterion(output_full, target)
                else:
                    loss = criterion(output, target) + criterion(output_full, target) + criterion_kl(output,
                                                                                                     output_full)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)

            if globals()['iterations'] % args.print_freq == 0:
                progress.print(i)

            end = time.time()

            # end of one mini-batch
            globals()['iterations'] += 1

        print('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        logging.info('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                        .format(top1=top1, top5=top5))

        return top1.avg, top5.avg

    def validate(self, args, val_loader, epoch, model, criterion):
        r"""Validate model each epoch and evaluation
        """
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                                 prefix='Test: ')

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (input, target, *other_info) in enumerate(val_loader):
                if args.device == 'cuda':
                    input = input.to(args.device, non_blocking=True)
                    target = target.to(args.device, non_blocking=True)
                if epoch < args.first_epoch:
                    output = model(input, 0, 0)
                else:
                    output = model(input, 0, 1)

                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
                top5.update(acc5[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)

                if i % args.print_freq == 0:
                    progress.print(i)

                end = time.time()

            print('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            logging.info('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                            .format(top1=top1, top5=top5))

        return top1.avg, top5.avg

    def validate_t(self, args, val_loader, epoch, model, criterion):
        r"""Validate model each epoch and evaluation
        """
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                                 prefix='Test: ')

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (input, target, *other_info) in enumerate(val_loader):
                if args.device == 'cuda':
                    input = input.to(args.device, non_blocking=True)
                    target = target.to(args.device, non_blocking=True)
                if epoch < args.first_epoch:
                    output = model(input, 1, 0)
                else:
                    output = model(input, 1, 1)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
                top5.update(acc5[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)

                if i % args.print_freq == 0:
                    progress.print(i)

                end = time.time()

            print('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            logging.info('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                         .format(top1=top1, top5=top5))

        return top1.avg, top5.avg

    def test(self, model, criterion, data_loader):
        model.eval()
        total_correct = 0
        total_loss = 0.0
        with torch.no_grad():
            for i, (images, labels, *other_info) in enumerate(data_loader):
                images, labels = images.to(args.device), labels.to(args.device)
                output = model(images, 4, 0)
                total_loss += criterion(output, labels).item()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
        loss = total_loss / len(data_loader)
        acc = float(total_correct) / len(data_loader.dataset)
        return loss, acc

    def test_clean_dataloader(self, args, double_model, test_dataloader, criterion, verbose = 0):
        double_model = double_model.to(args.device)
        criterion = criterion.to(args.device)
        double_model.eval()
        metrics = {
            'test_correct': 0,
            'test_loss_sum_over_batch': 0,
            'test_total': 0,
        }
        criterion = criterion.to(args.device, non_blocking=args.non_blocking)

        if verbose == 1:
            batch_predict_list, batch_label_list = [], []

        with torch.no_grad():
            for batch_idx, (x, target, *additional_info) in enumerate(test_dataloader):
                x = x.to(args.device, non_blocking=args.non_blocking)
                target = target.to(args.device, non_blocking=args.non_blocking)

                pred = double_model(x, 1, 1)
                loss = criterion(pred, target.long())

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                if verbose == 1:
                    batch_predict_list.append(predicted.detach().clone().cpu())
                    batch_label_list.append(target.detach().clone().cpu())

                metrics['test_correct'] += correct.item()
                metrics['test_loss_sum_over_batch'] += loss.item()
                metrics['test_total'] += target.size(0)

        metrics['test_loss_avg_over_batch'] = metrics['test_loss_sum_over_batch'] / len(test_dataloader)  ##########
        metrics['test_acc'] = metrics['test_correct'] / metrics['test_total']

        if verbose == 0:
            return metrics, None, None
        elif verbose == 1:
            return metrics, torch.cat(batch_predict_list), torch.cat(batch_label_list)

    def test_bd_dataloader_on_mix(self, args, double_model, test_dataloader, criterion, verbose = 0):
        double_model = double_model.to(args.device)
        double_model.eval()
        metrics = {
            'test_correct': 0,
            'test_loss_sum_over_batch': 0,
            'test_total': 0,
        }

        criterion = criterion.to(args.device, non_blocking=args.non_blocking)

        if verbose == 1:
            batch_predict_list = []
            batch_label_list = []
            batch_original_index_list = []
            batch_poison_indicator_list = []
            batch_original_targets_list = []
        with torch.no_grad():
            for batch_idx, (x, labels, original_index, poison_indicator, original_targets) in enumerate(
                    test_dataloader):
                x = x.to(args.device, non_blocking= args.non_blocking)
                labels = labels.to(args.device, non_blocking=args.non_blocking)
                pred = double_model(x, 1, 1)
                loss = criterion(pred, labels.long())

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(labels).sum()

                if verbose == 1:
                    batch_predict_list.append(predicted.detach().clone().cpu())
                    batch_label_list.append(labels.detach().clone().cpu())
                    batch_original_index_list.append(original_index.detach().clone().cpu())
                    batch_poison_indicator_list.append(poison_indicator.detach().clone().cpu())
                    batch_original_targets_list.append(original_targets.detach().clone().cpu())

                metrics['test_correct'] += correct.item()
                metrics['test_loss_sum_over_batch'] += loss.item()
                metrics['test_total'] += labels.size(0)

        metrics['test_loss_avg_over_batch'] = metrics['test_loss_sum_over_batch'] / len(test_dataloader)
        metrics['test_acc'] = metrics['test_correct'] / metrics['test_total']
        if verbose == 0:
            return metrics, \
                   None, None, None, None, None
        elif verbose == 1:
            return metrics, \
                   torch.cat(batch_predict_list), \
                   torch.cat(batch_label_list), \
                   torch.cat(batch_original_index_list), \
                   torch.cat(batch_poison_indicator_list), \
                   torch.cat(batch_original_targets_list)

    def evaluate_sNet(self, args, double_model, test_dataloader_dict):
        criterion = nn.CrossEntropyLoss()
        clean_metrics, \
        clean_test_epoch_predict_list, \
        clean_test_epoch_label_list, \
            = self.test_clean_dataloader(args, double_model, test_dataloader_dict["clean_test_dataloader"], criterion, verbose=1)

        clean_test_loss_avg_over_batch = clean_metrics["test_loss_avg_over_batch"]
        test_acc = clean_metrics["test_acc"]

        bd_metrics, \
        bd_test_epoch_predict_list, \
        bd_test_epoch_label_list, \
        bd_test_epoch_original_index_list, \
        bd_test_epoch_poison_indicator_list, \
        bd_test_epoch_original_targets_list \
            = self.test_bd_dataloader_on_mix(args, double_model, test_dataloader_dict["bd_test_dataloader"], criterion, verbose=1)


        bd_test_loss_avg_over_batch = bd_metrics["test_loss_avg_over_batch"]
        test_asr = all_acc(bd_test_epoch_predict_list, bd_test_epoch_label_list)
        test_ra = all_acc(bd_test_epoch_predict_list, bd_test_epoch_original_targets_list)

        return clean_test_loss_avg_over_batch, bd_test_loss_avg_over_batch, test_acc, test_asr, test_ra

    def set_Cleantrainer(self, model):
        trainer = ModelTrainerCLS_v2(
            model,
        )
        return trainer

    def set_devices(self):
        self.device = torch.device(
            (
                f"cuda:{[int(i) for i in self.args.device[5:].split(',')][0]}" if "," in self.args.device else self.args.device
                # since DataParallel only allow .to("cuda")
            ) if torch.cuda.is_available() else "cpu"
        )

    def reset(self, model):  # 重置模型中的参数
        for name, module in model.named_modules():
            if isinstance(module, MaskConv2d):
                module.reset()

    def change_structure(self, all_relevance, U_C):
        for i, layer in enumerate(all_relevance[0]):
            initial_size = layer.size()
            initial_size = initial_size[1:]
            initial_size = tuple(initial_size)
            U_C[i] = U_C[i].view(*initial_size)
        return U_C

    def revert_structure(self, U_C):
        for i, layer in enumerate(U_C):
            U_C[i] = layer.view(-1)
        return U_C

    def custom_union(self, array1, array2):
        # Create comparison condition
        arr = []
        for i in range(len(array1)):
            if array1[i] == array2[i]:
                arr.append(array1[i])
            else:
                arr.append(min(array1[i], array2[i]) + 1)
        return arr

    def transpose_nested_list(self, nested_list):

        first_level_len = len(nested_list)
        second_level_len = len(nested_list[0])

        # 创建新的嵌套列表
        new_nested_list = [[nested_list[j][i] for j in range(first_level_len)] for i in range(second_level_len)]

        return new_nested_list

    def set_feature_mask_for_myconv2d(self, module, feature_mask_values):
        if isinstance(module, MaskConv2d):
            data = feature_mask_values.pop().to(self.args.device, torch.float32)
            module.feature_m.data = data
            # 第一层mask保证为1

    def set_feature_mask_for_allconv2d(self, model, feature_mask_values):
        fmv = copy.deepcopy(feature_mask_values)

        if args.model == 'vgg':
            features_module = model.features
            for name, module in features_module.named_children():
                if isinstance(module, MaskConv2d):
                    self.set_feature_mask_for_myconv2d(module, fmv)
        elif args.model == 'resnet':
            features_module = model.features
            for name, module in features_module.named_children():
                if isinstance(module, MaskConv2d):
                    self.set_feature_mask_for_myconv2d(module, fmv)
        elif args.model == 'preactresnet':
            for name, module in model.named_children():
                if isinstance(module, MaskConv2d):
                    self.set_feature_mask_for_myconv2d(module, fmv)
                elif isinstance(module, nn.Sequential):
                    for name, sub_module in module.named_children():
                        premodule = sub_module.named_children()
                        for name, pre_module in premodule:
                            if isinstance(pre_module, MaskConv2d):
                                self.set_feature_mask_for_myconv2d(pre_module, fmv)
                            elif isinstance(pre_module, nn.Sequential):
                                for name, pre_sub_module in pre_module.named_children():
                                    if isinstance(pre_sub_module, MaskConv2d):
                                        self.set_feature_mask_for_myconv2d(pre_sub_module, fmv)
                            else:
                                continue
                else:
                    continue

    def evaluate_model(self, args, clean_test_dataset_with_trans, c_name):
        if c_name is None:
            c_name = args.save
        fixed_state_dict = torch.load(args.save_file + 'checkpoint/' + c_name)
        fixed_model = generate_cls_model(args.model_all, args.num_classes)
        load_state_dict_v2(fixed_model, fixed_state_dict)
        # main evaluation

        fixed_model.eval()
        train_tran = get_transform(args.dataset, *([args.input_height, args.input_width]), train=True)
        clean_dataset = prepro_cls_DatasetBD_v2(clean_test_dataset_with_trans.wrapped_dataset)
        data_set_without_tran = clean_dataset
        data_set_clean = self.result['clean_train']
        data_set_clean.wrapped_dataset = data_set_without_tran
        data_set_clean.wrap_img_transform = train_tran
        if args.dataset == 'cifar10':
            args.print_every = 100
        random_sampler = RandomSampler(data_source=data_set_clean, replacement=True,
                                       num_samples=args.print_every * args.batch_size)
        clean_val_loader = DataLoader(data_set_clean, batch_size=args.batch_size,
                                      shuffle=False, sampler=random_sampler, num_workers=args.num_workers)

        test_tran = get_transform(args.dataset, *([args.input_height, args.input_width]), train=False)
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        poison_test_loader = DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,
                                        drop_last=False, shuffle=True, pin_memory=True)
        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        clean_test_loader = DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,
                                       drop_last=False, shuffle=True, pin_memory=True)

        test_dataloader_dict = {}
        test_dataloader_dict["clean_test_dataloader"] = clean_test_loader
        test_dataloader_dict["bd_test_dataloader"] = poison_test_loader

        criterion = torch.nn.CrossEntropyLoss().to(args.device)
        trainer = set_PureCleantrainer(fixed_model)
        trainer.set_with_dataloader(
            ### the train_dataload has nothing to do with the backdoor defense
            train_dataloader=clean_val_loader,
            test_dataloader_dict=test_dataloader_dict,

            criterion=criterion,
            optimizer=None,
            scheduler=None,
            device=args.device,
            amp=args.amp,

            frequency_save=args.frequency_save,
            save_folder_path=args.save_file,
            save_prefix='dcil',

            prefetch=args.prefetch,
            prefetch_transform_attr_name="ori_image_transform_in_loading",
            non_blocking=args.non_blocking,

        )
        clean_test_loss_avg_over_batch, \
        bd_test_loss_avg_over_batch, \
        test_acc, \
        test_asr, \
        test_ra = trainer.test_current_model(
            test_dataloader_dict, args.device
        )
        print("clean_test_loss_avg_over_batch", clean_test_loss_avg_over_batch)
        print("bd_test_loss_avg_over_batch", bd_test_loss_avg_over_batch)
        print("test_acc", test_acc)
        print("test_asr", test_asr)
        print("test_ra", test_ra)
        logging.info("clean_test_loss_avg_over_batch:{}".format(clean_test_loss_avg_over_batch))
        logging.info("bd_test_loss_avg_over_batch:{}".format(bd_test_loss_avg_over_batch))
        logging.info("test_acc:{}".format(test_acc))
        logging.info("test_asr:{}".format(test_asr))
        logging.info("test_ra:{}".format(test_ra))

        if args.runtype == 'evaluate':
            agg = Metric_Aggregator()
            agg({
                "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                "test_acc": test_acc,
                "test_asr": test_asr,
                "test_ra": test_ra,
            })
            agg.to_dataframe().to_csv(f"{args.save_file}dcil_df_summary.csv")

            def evaluate_finetune_model(self, args, clean_test_dataset_with_trans, c_name):
                if c_name is None:
                    c_name = args.save
                fixed_state_dict = torch.load(args.save_file + 'checkpoint/' + c_name)
                fixed_model = generate_cls_model(args.model_all, args.num_classes)
                load_state_dict_v2(fixed_model, fixed_state_dict)
                # main evaluation

                fixed_model.eval()
                # a、prepare fixed_model
                train_tran = get_transform(args.dataset, *([args.input_height, args.input_width]), train=True)
                clean_dataset = prepro_cls_DatasetBD_v2(clean_test_dataset_with_trans.wrapped_dataset)
                data_set_without_tran = clean_dataset
                data_set_clean = self.result['clean_train']
                data_set_clean.wrapped_dataset = data_set_without_tran
                data_set_clean.wrap_img_transform = train_tran
                # data_set_clean.wrapped_dataset.getitem_all = False
                if args.dataset == 'gtsrb':
                    args.print_every = 100
                elif args.dataset == 'cifar10':
                    args.print_every = 100
                random_sampler = RandomSampler(data_source=data_set_clean, replacement=True,
                                               num_samples=args.print_every * args.batch_size)
                clean_val_loader = DataLoader(data_set_clean, batch_size=args.batch_size,
                                              shuffle=False, sampler=random_sampler, num_workers=args.num_workers)

                test_tran = get_transform(args.dataset, *([args.input_height, args.input_width]), train=False)
                data_bd_testset = self.result['bd_test']
                data_bd_testset.wrap_img_transform = test_tran
                poison_test_loader = DataLoader(data_bd_testset, batch_size=args.batch_size,
                                                num_workers=args.num_workers,
                                                drop_last=False, shuffle=True, pin_memory=True)
                data_clean_testset = self.result['clean_test']
                data_clean_testset.wrap_img_transform = test_tran
                clean_test_loader = DataLoader(data_clean_testset, batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=False, shuffle=True, pin_memory=True)

                test_dataloader_dict = {}
                test_dataloader_dict["clean_test_dataloader"] = clean_test_loader
                test_dataloader_dict["bd_test_dataloader"] = poison_test_loader

                criterion = torch.nn.CrossEntropyLoss().to(args.device)
                trainer = set_PureCleantrainer(fixed_model)
                trainer.set_with_dataloader(
                    train_dataloader=clean_val_loader,
                    test_dataloader_dict=test_dataloader_dict,

                    criterion=criterion,
                    optimizer=None,
                    scheduler=None,
                    device=args.device,
                    amp=args.amp,

                    frequency_save=args.frequency_save,
                    save_folder_path=args.save_file,
                    save_prefix='dcil',

                    prefetch=args.prefetch,
                    prefetch_transform_attr_name="ori_image_transform_in_loading",
                    non_blocking=args.non_blocking,

                )
                clean_test_loss_avg_over_batch, \
                bd_test_loss_avg_over_batch, \
                test_acc, \
                test_asr, \
                test_ra = trainer.test_current_model(
                    test_dataloader_dict, args.device
                )
                print("clean_test_loss_avg_over_batch", clean_test_loss_avg_over_batch)
                print("bd_test_loss_avg_over_batch", bd_test_loss_avg_over_batch)
                print("test_acc", test_acc)
                print("test_asr", test_asr)
                print("test_ra", test_ra)
                logging.info("clean_test_loss_avg_over_batch:{}".format(clean_test_loss_avg_over_batch))
                logging.info("bd_test_loss_avg_over_batch:{}".format(bd_test_loss_avg_over_batch))
                logging.info("test_acc:{}".format(test_acc))
                logging.info("test_asr:{}".format(test_asr))
                logging.info("test_ra:{}".format(test_ra))
                if args.runtype == 'evaluate':
                    agg = Metric_Aggregator()
                    agg({
                        "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                        "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                        "test_acc": test_acc,
                        "test_asr": test_asr,
                        "test_ra": test_ra,
                    })
                    agg.to_dataframe().to_csv(f"{args.save_file}dcil_df_summary.csv")
                    logging.info(f"save to {args.save_file}dcil_df_summary.csv")

    def lrp_path(self):
        self.set_devices()
        fix_random(self.args.random_seed)

        args = self.args
        # prepare model
        if args.attack == False:
            state_dict = self.result
        else:
            state_dict = self.result['model']
        model = generate_cls_model(args.model_all, args.num_classes)
        model.load_state_dict(state_dict)
        model.to(args.device)
        model.eval()
        if 'preactresnet' in args.model:
            lrp_model = lrp.convert_preactresnet(model).to(args.device)
        elif 'resnet' in args.model:
            lrp_model = lrp.convert_resnet(model).to(args.device)
        elif 'vgg' in args.model:
            lrp_model = lrp.convert_vgg(model).to(args.device)
        else:
            raise NotImplementedError
        # prepare data
        if args.attack == False:
            train_dataset_without_transform, \
            train_img_transform, \
            train_label_transform, \
            test_dataset_without_transform, \
            test_img_transform, \
            test_label_transform = dataset_and_transform_generate(args)

            clean_train_dataset_with_trans = dataset_wrapper_with_transform(
                train_dataset_without_transform,
                train_img_transform,
                train_label_transform,
            )
            clean_test_dataset_with_trans = dataset_wrapper_with_transform(
                test_dataset_without_transform,
                test_img_transform,
                test_label_transform,
            )
        else:
            clean_train_dataset_with_trans = self.result['clean_train']
            clean_test_dataset_with_trans = self.result['clean_test']

        train_tran = get_transform_extends(args, self.args.dataset, *([self.args.input_height, self.args.input_width]),
                                           train=True)
        test_tran = get_transform_extends(args, self.args.dataset, *([self.args.input_height, self.args.input_width]),
                                          train=False)
        clean_train = prepro_cls_DatasetBD_v2(clean_train_dataset_with_trans.wrapped_dataset)
        clean_test = prepro_cls_DatasetBD_v2(clean_test_dataset_with_trans.wrapped_dataset)
        data_all_length1 = len(clean_train)
        data_all_length2 = len(clean_test)
        ran_idx1 = choose_index(self.args, data_all_length1)
        ran_idx2 = choose_index(self.args, data_all_length2)
        log_index1 = self.args.log + 'index_train.txt'
        log_index2 = self.args.log + 'index_test.txt'
        np.savetxt(log_index1, ran_idx1, fmt='%d')
        np.savetxt(log_index2, ran_idx2, fmt='%d')
        clean_train.subset(ran_idx1)
        clean_test.subset(ran_idx2)
        train_data_set_without_tran = clean_train
        test_data_set_without_tran = clean_test
        train_data_set_o = clean_train_dataset_with_trans
        test_data_set_o = clean_test_dataset_with_trans
        train_data_set_o.wrapped_dataset = train_data_set_without_tran
        test_data_set_o.wrapped_dataset = test_data_set_without_tran
        train_data_set_o.wrap_img_transform = train_tran
        test_data_set_o.wrap_img_transform = test_tran
        train_loader = torch.utils.data.DataLoader(train_data_set_o, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, shuffle=False,
                                                   pin_memory=args.pin_memory)
        val_loader = torch.utils.data.DataLoader(test_data_set_o, batch_size=args.batch_size,
                                                 num_workers=args.num_workers, shuffle=False,
                                                 pin_memory=args.pin_memory)

        test_tran_1 = get_transform(self.args.dataset, *([self.args.input_height, self.args.input_width]), train=False)
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran_1
        # data_bd_testset.wrapped_dataset.getitem_all = False
        poison_test_loader = DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,
                                        drop_last=False, shuffle=True, pin_memory=True)

        test_tran_2 = get_transform(self.args.dataset, *([self.args.input_height, self.args.input_width]), train=False)
        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran_2
        clean_test_loader = DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,
                                       drop_last=False, shuffle=True, pin_memory=True)
        test_dataloader_dict = {}
        test_dataloader_dict["clean_test_dataloader"] = clean_test_loader
        test_dataloader_dict["bd_test_dataloader"] = poison_test_loader

        file_name = os.path.join(args.save_path, 'utimate_c.pth')
        file_name1 = os.path.join(args.save_path, 'all_relevance.pth')


        if not os.path.exists(file_name) or not os.path.exists(file_name1):
            for idx, (x_batch, y_batch, *other_info) in enumerate(train_loader):
                x_batch = x_batch.to(args.device)
                y_batch = y_batch.to(args.device)

                if idx == 0:
                    x_extra, y_extra = x_batch, y_batch
                else:
                    x_extra = torch.cat((x_extra, x_batch))
                    y_extra = torch.cat((y_extra, y_batch))
                if idx > 10:
                    break

            num_samples = 30
            for i in range(args.num_classes):
                # size = np.count_nonzero(y_extra == i)
                size = torch.count_nonzero(y_extra == i).item()
                if size < num_samples:
                    num_samples = size
            assert (num_samples > 0)

            indices = []
            for i in range(args.num_classes):
                # idx = np.where(y_extra == i)[0]
                idx = torch.where(y_extra == i)[0]
                idx = idx.detach().cpu().numpy()
                indices.extend(list(idx[:num_samples]))
            x_extra = x_extra[indices]
            y_extra = y_extra[indices]
            assert (x_extra.size(0) == num_samples * args.num_classes)

            all_relevance = [[] for i in range(args.num_classes)]
            # # # # # Patterns for PatternNet and PatternAttribution
            from pathlib import Path
            patterns_path = (Path(args.save_path) / 'examples' / 'patterns' / ('vgg%i_pattern_pos.pkl' % 16)).as_posix()

            if not os.path.exists(patterns_path):
                patterns = fit_patternnet_positive(lrp_model, train_loader, device=args.device)
                # patterns = fit_patternnet(lrp_model, trainloader, device=args.device)
                store_patterns(patterns_path, patterns)
            else:
                patterns = load_patterns(patterns_path, device=args.device)
            print("Loaded patterns")
            logging.info("Loaded patterns")
            for n_class in range(args.num_classes):
                x_batch = x_extra[n_class * num_samples:(n_class + 1) * num_samples].to(args.device)
                y_batch = y_extra[n_class * num_samples:(n_class + 1) * num_samples].to(args.device)

                # true label
                x_batch.requires_grad_(True)
                trace.enable_and_clean()
                y_ = model(x_batch)

                rule = 'patternattribution'
                patterns = patterns
                y_hat_lrp = lrp_model.forward(x_batch, explain=True, rule=rule, pattern=patterns)

                y_hat_lrp = y_hat_lrp[torch.arange(num_samples), y_hat_lrp.max(1)[1]]  # Choose maximizing output neuron
                y_hat_lrp1 = y_hat_lrp.sum()

                # Backward pass (do explanation)
                y_hat_lrp1.backward()
                layer_relevances = trace.collect_and_disable()
                explanation = x_batch.grad
                all_relevance[n_class] = layer_relevances
            all_re = copy.deepcopy(all_relevance)
            args.ns = 0.25
            relevance_classes = [[] for i in range(args.num_classes)]
            for k in range(args.num_classes):
                layer_relevance = all_re[k].copy()
                relevance_class = [[] for i in range(len(layer_relevance))]
                for layer in range(len(layer_relevance)):
                    new_M = layer_relevance[layer].numel() // num_samples
                    relevance_samples = layer_relevance[layer].view(num_samples, new_M)
                    for idx, row_relavance in enumerate(relevance_samples):
                        row_grade = torch.sort(row_relavance, descending=True)[1]
                        row_relavance[row_grade[0:int(new_M * args.ns)]] = 1
                        row_relavance[row_grade[int(new_M * args.ns):int(new_M * 2 * args.ns)]] = 2  # critical
                        row_relavance[row_grade[int(new_M * 2 * args.ns):]] = 0

                    rs_layer = torch.zeros(new_M)
                    for lie, lie_number in enumerate(relevance_samples.T):
                        lie_number = lie_number.detach().cpu().numpy()
                        unimportant = np.count_nonzero(lie_number == 0)
                        critical = np.count_nonzero(lie_number == 2)
                        important = np.count_nonzero(lie_number == 1)
                        if unimportant >= num_samples / 2:
                            rs_layer[lie] = 0
                        elif critical >= important:
                            rs_layer[lie] = 2
                        else:
                            rs_layer[lie] = 1
                    rs_layer = rs_layer.numpy().tolist()
                    relevance_class[layer] = rs_layer
                relevance_classes[k] = relevance_class
            rc_trans = self.transpose_nested_list(relevance_classes)
            utimate_c = []
            for layer_r, r_l in enumerate(rc_trans):
                for k in range(1, len(r_l)):
                    r_l_k = np.array(r_l[k])
                    r_l_k_1 = np.array(r_l[k - 1])
                    r_l_utimate = self.custom_union(r_l_k, r_l_k_1)
                    if k == len(r_l) - 1:
                        utimate_c.append(torch.tensor(r_l_utimate))
                    # r_l[k] = torch.tensor(r_l_utimate)

            print("Finished neuron ranking")
            logging.info("Finished neuron ranking")
            torch.save(utimate_c, file_name)
            torch.save(all_relevance, file_name1)

        pruner = PA.__dict__[args.pruner]
        double_model = PA.PA_models.__dict__[args.model](num_layers=args.layers,
                                                         num_classes=args.num_classes,
                                                         width_mult=args.width_mult,
                                                         depth_mult=args.depth_mult,
                                                         model_mult=args.model_mult,
                                                         mnn=pruner.mnn)
        load_state_dict(double_model, state_dict)
        double_model.to(self.args.device)
        double_model.eval()
        assert double_model is not None, 'Unavailable model parameters!! exit...\n'

        if args.runtype == 'train':
            U_C = torch.load(file_name)
            all_relevance = torch.load(file_name1)
            criterion = nn.CrossEntropyLoss()
            if args.mode_type != 'Ablation1':
                print('The mode type is normal, need to search key neurons')
                logging.info('The mode type is normal, need to search key neurons')
                critical_neurons = []
                key_neurons = []
                for i, tensor in enumerate(U_C):
                    for j in range(len(U_C[i])):
                        if i == 0 or i == len(U_C) - 1 or i == len(U_C) - 2:
                            # if i == len(U_C) - 1:
                            U_C[i][j] = 2
                        if U_C[i][j] == 2:  # critical
                            critical_neurons.append((i, j))
                        elif U_C[i][j] == 1:  # important
                            key_neurons.append((i, j))
                    U_C[i] = torch.where(tensor > 1, torch.ones_like(tensor), torch.zeros_like(tensor))

                print("Finished neuron selection")
                logging.info("Finished neuron selection")

                utimate_copy = copy.deepcopy(U_C)
                utimate_copy = self.change_structure(all_relevance, utimate_copy)
                self.set_feature_mask_for_allconv2d(double_model, utimate_copy)

                cl_loss, cl_acc = self.test(model=double_model, criterion=criterion, data_loader=train_loader)
                print('0 \t      {:.4f} \t {:.4f}'.format(cl_loss, cl_acc))
                logging.info('0 \t       {:.4f} \t {:.4f}'.format(cl_loss, cl_acc))
                sample_neuron_sum = 0
                max_acc = 0
                print("Start sampling")
                logging.info("Start sampling")
                COUNT = 0
                len_k = len(key_neurons)  # key_neurons的最初长度
                while max_acc < args.MODEL_ACC * 1 / 5:
                    acc_candidates = []
                    candidates_neurons = []
                    utimate_c_n = []

                    print('key_neurons:{0}', len(key_neurons))
                    COUNT += 1
                    for s_f in range(0, args.sample_f):
                        utimate_copy = copy.deepcopy(U_C)
                        critical_neurons_candidates = random.sample(key_neurons, int(len(key_neurons) * args.sample_r))
                        for candidates in critical_neurons_candidates:
                            (i, j) = candidates
                            utimate_copy[i][j] = 1
                        utimate_copy = self.change_structure(all_relevance, utimate_copy)
                        self.set_feature_mask_for_allconv2d(double_model, utimate_copy)
                        utimate_copy = self.revert_structure(utimate_copy)
                        cl_loss, cl_acc = self.test(model=double_model, criterion=criterion, data_loader=train_loader)
                        print('------sample frequency{}------'.format(s_f))
                        print('0  \t {:.4f} \t {:.4f}'.format(cl_loss, cl_acc))
                        logging.info('------sample frequency{}------'.format(s_f))
                        logging.info('0 \t  {:.4f} \t {:.4f}'.format(cl_loss, cl_acc))
                        candidates_neurons.append(critical_neurons_candidates)
                        acc_candidates.append(cl_acc)
                        utimate_c_n.append(utimate_copy)
                    max_acc = max(acc_candidates)
                    location = np.argmax(acc_candidates)
                    U_C = utimate_c_n[location]
                    critical_n = candidates_neurons[location]
                    if COUNT > 8:
                        break
                    sample_neuron_sum += len(critical_n)
                    print('sample_neuron_sum:', sample_neuron_sum)
                    logging.info('sample_neuron_sum:{}'.format(sample_neuron_sum))

                    for candidates in critical_n:
                        key_neurons.remove(candidates)
                        critical_neurons.append(candidates)

                print(f'max_acc:{max_acc}')
                print('Finished sampling')
                logging.info(f'max_acc:{max_acc}')
                logging.info('Finished sampling')
                print('sample_neuron_sum/key_neuron_sum:', sample_neuron_sum / len_k)
                logging.info('sample_neuron_sum/key_neuron_sum:{}'.format(sample_neuron_sum / len_k))


                self.change_structure(all_relevance, U_C)
                self.set_feature_mask_for_allconv2d(double_model, U_C)

            self.change_structure(all_relevance, U_C)
            self.set_feature_mask_for_allconv2d(double_model, U_C)

            if args.mode_type == 'Ablation2':
                args.initial_cov_rate = 1.0


            # set criterion and optimizer
            optimizer = optim.SGD(double_model.parameters(), lr=args.lr,
                                  momentum=args.momentum, weight_decay=args.weight_decay,
                                  nesterov=args.nesterov)
            scheduler = set_scheduler(optimizer, args)

            print('==> Start training')
            logging.info('==> Start training')
            # init parameters
            global iterations
            iterations = 0
            best_acc1 = 0.0
            train_time = 0.0
            validate_time = 0.0


            # for epoch in range(start_epoch, args.epochs):
            for epoch in range(0, args.target_epoch):
                logging.info('\n==> {}/{} training'.format(
                    args.model, args.dataset))
                logging.info('==> Epoch: {}, lr = {}'.format(
                    epoch, optimizer.param_groups[0]["lr"]))

                # train for one epoch
                logging.info('===> [ Training ]')
                start_time = time.time()

                if epoch < args.first_epoch:
                    acc1_train, acc5_train = self.train_first(args, train_loader,
                                                              epoch=epoch, model=double_model,
                                                              criterion=criterion, optimizer=optimizer,
                                                              scheduler=scheduler)
                else:
                    acc1_train, acc5_train = self.train_second(args, train_loader,
                                                               epoch=epoch, model=double_model,
                                                               criterion=criterion, optimizer=optimizer,
                                                               scheduler=scheduler)

                elapsed_time = time.time() - start_time
                train_time += elapsed_time
                logging.info('====> {:.2f} seconds to train this epoch\n'.format(
                    elapsed_time))

                # evaluate on validation set
                logging.info('===> [ Validation ]')
                start_time = time.time()
                acc1_valid, acc5_valid = self.validate(args, val_loader,
                                                       epoch=epoch, model=double_model, criterion=criterion)
                elapsed_time = time.time() - start_time
                validate_time += elapsed_time
                logging.info('====> {:.2f} seconds to validate this epoch'.format(
                    elapsed_time))

                tt1, tt = self.validate_t(args, val_loader,
                                          epoch=epoch, model=double_model, criterion=criterion)

                acc1_train = round(acc1_train.item(), 4)
                acc5_train = round(acc5_train.item(), 4)
                acc1_valid = round(acc1_valid.item(), 4)
                acc5_valid = round(acc5_valid.item(), 4)


                # remember best Acc@1 and save checkpoint and summary csv file
                state = double_model.state_dict()
                summary = [epoch, acc1_train, acc5_train, acc1_valid, acc5_valid]

                if epoch == args.first_epoch:
                    save_model(args, state, 'finetune.pth')
                else:
                    save_model(args, state, args.save)
                save_summary(args, args.model, args.dataset, args.save.split('.pth')[0], summary)

                # for pruning
                num_total, mask_nonzeros, sparsity = PA.cal_sparsity(double_model)
                logging.info('\n====> sparsity: {:.2f}% || mask_nonzeros/num_total: {}/{}'.format(sparsity, mask_nonzeros,
                                                                                           num_total))

            avg_train_time = train_time / (args.target_epoch - 0)
            avg_valid_time = validate_time / (args.target_epoch - 0)
            total_train_time = train_time + validate_time
            print('====> average training time each epoch: {:,}m {:.2f}s'.format(
                int(avg_train_time // 60), avg_train_time % 60))
            print('====> average validation time each epoch: {:,}m {:.2f}s'.format(
                int(avg_valid_time // 60), avg_valid_time % 60))
            print('====> training time: {}h {}m {:.2f}s'.format(
                int(train_time // 3600), int((train_time % 3600) // 60), train_time % 60))
            print('====> validation time: {}h {}m {:.2f}s'.format(
                int(validate_time // 3600), int((validate_time % 3600) // 60), validate_time % 60))
            print('====> total training time: {}h {}m {:.2f}s'.format(
                int(total_train_time // 3600), int((total_train_time % 3600) // 60), total_train_time % 60))

            logging.info('====>Finetune Evaluation<=====')
            logging.info('====> average training time each epoch: {:,}m {:.2f}s'.format(
                int(avg_train_time // 60), avg_train_time % 60))
            logging.info('====> average validation time each epoch: {:,}m {:.2f}s'.format(
                int(avg_valid_time // 60), avg_valid_time % 60))
            logging.info('====> training time: {}h {}m {:.2f}s'.format(
                int(train_time // 3600), int((train_time % 3600) // 60), train_time % 60))
            logging.info('====> validation time: {}h {}m {:.2f}s'.format(
                int(validate_time // 3600), int((validate_time % 3600) // 60), validate_time % 60))
            logging.info('====> total training time: {}h {}m {:.2f}s'.format(
                int(total_train_time // 3600), int((total_train_time % 3600) // 60), total_train_time % 60))

            logging.info('====>Finetune Evaluation<=====')

            if args.attack:
                self.evaluate_model(args, clean_test_dataset_with_trans, 'finetune.pth')

            # for evaluation on validation set
            print('\n===> [ Final Evaluation ]')
            logging.info('===> [ Final Evaluation ]')
            agg = Metric_Aggregator()
            clean_test_loss_avg_over_batch, \
            bd_test_loss_avg_over_batch,\
            test_acc, \
            test_asr, \
            test_ra = self.evaluate_sNet(args, double_model, test_dataloader_dict)
            agg({
                "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                "test_acc": test_acc,
                "test_asr": test_asr,
                "test_ra": test_ra,
            })
            logging.info(agg)
            agg.to_dataframe().to_csv(f"{args.save_file}sparse_network_df_summary.csv")

        elif args.runtype == 'evaluate':  # for evaluation
            # for evaluation on validation set
            print('\n===> [ Final Evaluation ]')
            start_time = time.time()
            if args.attack:
                self.evaluate_model(args, clean_test_dataset_with_trans, None)
            elapsed_time = time.time() - start_time
            print('====> {:.2f} seconds to evaluate this model\n'.format(elapsed_time))

        else:
            assert False, 'Unkown --run-type! It should be \{train, evaluate\}.'

    def defense(self, result_file):
        self.set_result(result_file)
        self.set_logger()
        self.lrp_path()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    LRP_path_dete.add_arguments(parser)
    args = parser.parse_args()
    lrp_method = LRP_path_dete(args)
    lrp_method.defense(args.result_file)
