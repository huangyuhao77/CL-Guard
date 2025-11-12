import argparse
import sys
import os


sys.path.append('../')
sys.path.append(os.getcwd())
from torch.utils.data import RandomSampler, DataLoader
from utils.trainer_cls import PureCleanModelTrainer, Metric_Aggregator
from utils.model_trainer_generate import generate_cls_model
import time
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch import nn
import torch.nn.functional as F

from torch.autograd import Variable

import warnings

import PA
from PA import MaskConv2d, load_state_dict
from run_prune_utils import accuracy, AverageMeter, save_summary, set_scheduler, save_model, ProgressMeter, load_state_dict_v2
from utils.dataset_and_transform_generate import get_num_classes, get_input_shape, get_transform, \
    dataset_and_transform_generate
from utils.bd_dataset_v2 import dataset_wrapper_with_transform, prepro_cls_DatasetBD_v2
from utils.choose_index import choose_index
from utils.save_load_attack import load_attack_result

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
warnings.filterwarnings("ignore", "(Palette )?images with Transparency", UserWarning)


def set_result(args, result_file):
    attack_file = './record/' + result_file
    save_file = './record/' + result_file + '/defense/dcil_finetune/'
    npy_save_file = './record/' + result_file + '/defense/lrp_path/checkpoint/'
    args.npy_save_file = npy_save_file
    if not (os.path.exists(save_file)):
        os.makedirs(save_file)
    args.save_file = save_file
    log = save_file + 'log/'
    if not (os.path.exists(log)):
        os.makedirs(log)
    args.log = log
    if args.attack == False:
        result = torch.load(attack_file + '/clean_model.pth')
    else:
        result = load_attack_result(attack_file + '/attack_result.pt')
    return result

class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T = 2

        predict =F.log_softmax(pred / T, dim=1)
        target_data = F.softmax(label / T, dim=1)
        target_data = target_data + 10 ** (-7)
        target = Variable(target_data.data.to(args.device), requires_grad=False)
        loss = T * T * ((target * (target.log() - predict)).sum(1).sum() / target.size()[0])
        return loss


criterion_kl = KLLoss().cuda()


def hyperparam():
    parser = argparse.ArgumentParser(description=sys.argv[0])
    import configition
    args = configition.config(parser)
    return args


def set_trainer(model):
    trainer = PureCleanModelTrainer(
        model,
    )
    return trainer


def main(args):
    if not torch.cuda.is_available():
        raise Exception('No GPU found, please run without --cuda')

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.cu_num

    # set model name
    print('\n=> creating model \'{}\''.format(args.model))
    args.num_classes = get_num_classes(args.dataset)
    args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
    args.img_size = (args.input_height, args.input_width, args.input_channel)
    args.dataset_path = f"{args.dataset_path}/{args.dataset}"
    result = set_result(args, args.result_file)
    # result为原始模型

    if args.prune:  # for pruning
        pruner = PA.__dict__[args.pruner]
        model = PA.PA_models.__dict__[args.model](num_layers=args.layers,
                                              num_classes = args.num_classes,
                                                            width_mult=args.width_mult,
                                                            depth_mult=args.depth_mult,
                                                            model_mult=args.model_mult,
                                                            mnn=pruner.mnn)
    assert model is not None, 'Unavailable model parameters!! exit...\n'

    # set criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay,
                          nesterov=args.nesterov)
    scheduler = set_scheduler(optimizer, args)

    # set multi-gpu
    if args.device == 'cuda' and torch.cuda.is_available():
        model = model.to(args.device)
        criterion = criterion.to(args.device)
        # model = nn.DataParallel(model, device_ids=args.gpuids,
        #                         output_device=args.gpuids[0])
        cudnn.benchmark = True

        # for distillation

    # Dataset loading
    print('==> Load data..')
    start_time = time.time()

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
        clean_train_dataset_with_trans = result['clean_train']
        clean_test_dataset_with_trans = result['clean_test']
    train_tran = get_transform(args.dataset, *([args.input_height,args.input_width]),
                               train=True)
    clean_train_dataset = prepro_cls_DatasetBD_v2(clean_train_dataset_with_trans.wrapped_dataset)
    clean_test_dataset = prepro_cls_DatasetBD_v2(clean_test_dataset_with_trans.wrapped_dataset)
    data1_all_length = len(clean_train_dataset)
    data2_all_length = len(clean_test_dataset)
    ran_idx1 = choose_index(args, data1_all_length)
    ran_idx2 = choose_index(args, data2_all_length)
    log_index1 = args.log + 'index1.txt'
    log_index2 = args.log + 'index2.txt'
    np.savetxt(log_index1, ran_idx1, fmt='%d')
    np.savetxt(log_index2, ran_idx2, fmt='%d')
    clean_train_dataset.subset(ran_idx1)
    clean_test_dataset.subset(ran_idx2)
    train_data_set_without_tran = clean_train_dataset
    test_data_set_without_tran = clean_test_dataset

    train_data_set_o = clean_train_dataset_with_trans
    test_data_set_o = clean_test_dataset_with_trans
    train_data_set_o.wrapped_dataset = train_data_set_without_tran
    test_data_set_o.wrapped_dataset = test_data_set_without_tran
    train_data_set_o.wrap_img_transform = train_tran
    test_data_set_o.wrap_img_transform = train_tran
    train_loader = torch.utils.data.DataLoader(train_data_set_o, batch_size=args.batch_size,
                                              num_workers=args.workers, shuffle=False, pin_memory=args.pin_memory)
    val_loader = torch.utils.data.DataLoader(test_data_set_o, batch_size=args.batch_size,
                                            num_workers=args.workers, shuffle=False, pin_memory=args.pin_memory)


    elapsed_time = time.time() - start_time
    print('===> Data loading time: {:,}m {:.2f}s'.format(
        int(elapsed_time // 60), elapsed_time % 60))
    print('===> Data loaded..')

    # load a pre-trained model
    if args.attack == False:
        state_dict = result
    else:
        state_dict = result['model']
    print('==> Loading Checkpoint \'{}\''.format(args.result_file))
    # check pruning or quantization or transfer
    # strict = False if args.prune else True
    # load a checkpoint

    print('==> Loaded Checkpoint \'{}\''.format(args.result_file))

    # for training
    if args.run_type == 'train':
        load_state_dict(model, state_dict)
        # init parameters
        global iterations
        iterations = 0
        best_acc1 = 0.0
        train_time = 0.0
        validate_time = 0.0

        os.makedirs('./results', exist_ok=True)
        file_train_acc = os.path.join('results', '{}.txt'.format(
            '_'.join(['train', args.model, args.dataset, args.save.split('.pth')[0]])))
        file_test_acc = os.path.join('results', '{}.txt'.format(
            '_'.join(['test', args.model, args.dataset, args.save.split('.pth')[0]])))

        # for epoch in range(start_epoch, args.epochs):
        for epoch in range(0, args.target_epoch):
            print('\n==> {}/{} training'.format(
                args.model, args.dataset))
            print('==> Epoch: {}, lr = {}'.format(
                epoch, optimizer.param_groups[0]["lr"]))

            # train for one epoch
            print('===> [ Training ]')
            start_time = time.time()
            acc1_train, acc5_train = train(args, train_loader,
                                           epoch=epoch, model=model,
                                           criterion=criterion, optimizer=optimizer, scheduler=scheduler)

            elapsed_time = time.time() - start_time
            train_time += elapsed_time
            print('====> {:.2f} seconds to train this epoch\n'.format(
                elapsed_time))

            # evaluate on validation set
            print('===> [ Validation ]')
            start_time = time.time()
            acc1_valid, acc5_valid = validate(args, val_loader,
                                              epoch=epoch, model=model, criterion=criterion)
            elapsed_time = time.time() - start_time
            validate_time += elapsed_time
            print('====> {:.2f} seconds to validate this epoch'.format(
                elapsed_time))

            tt1, tt = validate_t(args, val_loader,
                                 epoch=epoch, model=model, criterion=criterion)

            acc1_train = round(acc1_train.item(), 4)
            acc5_train = round(acc5_train.item(), 4)
            acc1_valid = round(acc1_valid.item(), 4)
            acc5_valid = round(acc5_valid.item(), 4)

            open(file_train_acc, 'a').write(str(acc1_train) + '\n')
            open(file_test_acc, 'a').write(str(acc1_valid) + '\n')

            # remember best Acc@1 and save checkpoint and summary csv file
            state = model.state_dict()
            summary = [epoch, acc1_train, acc5_train, acc1_valid, acc5_valid]

            # is_best = acc1_valid > best_acc1
            # best_acc1 = max(acc1_valid, best_acc1)
            # if is_best:
            save_model(args, state, args.save)
            save_summary(args, args.model, args.dataset, args.save.split('.pth')[0], summary)

            # for pruning
            if args.prune:
                num_total, mask_nonzeros, sparsity = PA.cal_sparsity(model)
                print('\n====> sparsity: {:.2f}% || mask_nonzeros/num_total: {}/{}'.format(sparsity, mask_nonzeros, num_total))

            # end of one epoch
            print()

        # calculate the total training time
        # avg_train_time = train_time / (args.epochs - start_epoch)
        # avg_valid_time = validate_time / (args.epochs - start_epoch)
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

        return best_acc1

    elif args.run_type == 'evaluate':  # for evaluation
        # for evaluation on validation set
        print('\n===> [ Evaluation ]')
        fixed_state_dict = torch.load(args.save_file + 'checkpoint/' + "ckpt.pth")
        fixed_model = generate_cls_model(args.model_all, args.num_classes)
        load_state_dict_v2(fixed_model, fixed_state_dict)
        # main evaluation
        start_time = time.time()
        # 测评该模型是否仍有后门
        fixed_model.eval()
        # a、prepare fixed_model
        train_tran = get_transform(args.dataset, *([args.input_height, args.input_width]), train=True)
        clean_dataset = prepro_cls_DatasetBD_v2(clean_test_dataset_with_trans.wrapped_dataset)
        data_set_without_tran = clean_dataset
        data_set_clean = result['clean_train']
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
                                      shuffle=False, sampler=random_sampler, num_workers=args.workers)

        test_tran = get_transform(args.dataset, *([args.input_height, args.input_width]), train=False)
        data_bd_testset = result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        # data_bd_testset.wrapped_dataset.getitem_all = False
        poison_test_loader = DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.workers,
                                        drop_last=False, shuffle=True, pin_memory=True)
        data_clean_testset = result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        clean_test_loader = DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.workers,
                                       drop_last=False, shuffle=True, pin_memory=True)

        test_dataloader_dict = {}
        test_dataloader_dict["clean_test_dataloader"] = clean_test_loader
        test_dataloader_dict["bd_test_dataloader"] = poison_test_loader

        # 两个网络的ASR和ACC
        criterion = torch.nn.CrossEntropyLoss().to(args.device)
        trainer = set_trainer(fixed_model)
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
        agg = Metric_Aggregator()
        clean_test_loss_avg_over_batch, \
        bd_test_loss_avg_over_batch, \
        test_acc, \
        test_asr, \
        test_ra = trainer.test_current_model(
            test_dataloader_dict, args.device
        )
        agg({
            "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
            "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
            "test_acc": test_acc,
            "test_asr": test_asr,
            "test_ra": test_ra,
        })
        print("clean_test_loss_avg_over_batch", clean_test_loss_avg_over_batch)
        print("bd_test_loss_avg_over_batch", bd_test_loss_avg_over_batch)
        print("test_acc", test_acc)
        print("test_asr", test_asr)
        print("test_ra", test_ra)
        agg.to_dataframe().to_csv(f"{args.save_file}dcil_df_summary.csv")
        elapsed_time = time.time() - start_time
        print('====> {:.2f} seconds to evaluate this model\n'.format(elapsed_time))
    else:
        assert False, 'Unkown --run-type! It should be \{train, evaluate\}.'

def set_feature_mask_for_myconv2d(module, feature_mask_values):
    print(isinstance(module, MaskConv2d))
    print(module)
    if isinstance(module, MaskConv2d):
        data = feature_mask_values.pop()
        module.feature_m.data = data
    #第一层mask保证为1

def train(args, train_loader, epoch, model, criterion, optimizer, scheduler, **kwargs):
    r"""Train model each epoch
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time,
                             losses, top1, top5, prefix="Epoch: [{}]".format(epoch))
    if epoch == 0:# 第一次训练，首先要将寻找到的一批神经元传入mask中
        feature_mask_values = []
        union_class_neuron = torch.load(str(args.npy_save_file)+'union_class.pt')
        for i, conv2d_neuron in enumerate(union_class_neuron):
            if args.model == 'vgg':
                if i > (len(union_class_neuron) - 1) / 3 * 2: # 16层 后五层需要
                    feature_mask_values.append(torch.ones_like(union_class_neuron[i]))
                else:
                    feature_mask_values.append(conv2d_neuron)
            elif args.model == 'resnet':
                if i >= (len(union_class_neuron)-1) / 3 * 2:
                    feature_mask_values.append(torch.ones_like(union_class_neuron[i]))
                else:
                    feature_mask_values.append(conv2d_neuron)

            elif args.model == 'preactresnet':
                if i ==0 or i == 1:
                    feature_mask_values.append(torch.ones_like(union_class_neuron[i]))
                elif i >= (len(union_class_neuron)-1) / 3 * 2:
                    feature_mask_values.append(torch.ones_like(union_class_neuron[i]))
                else:
                    feature_mask_values.append(conv2d_neuron)
            else:
                raise NotImplementedError
        # Apply different mask values to different MyConv2d layers
        if args.model == 'vgg':
            # 获取 VGG 模型的 features 模块
            features_module = model.features

            # 遍历 features 模块中的子模块
            for name, module in features_module.named_children():
                # 判断当前子模块是否为 MaskConv2d
                if isinstance(module, MaskConv2d):
                    # 对MaskConv2d 模块执行 set_mask_for_myconv2d 操作
                    set_feature_mask_for_myconv2d(module, feature_mask_values)
        elif args.model == 'resnet':
            # 获取 VGG 模型的 features 模块
            features_module = model.features

            # 遍历 features 模块中的子模块
            for name, module in features_module.named_children():
                # 判断当前子模块是否为 MaskConv2d
                if isinstance(module, MaskConv2d):
                    # 对MaskConv2d 模块执行 set_mask_for_myconv2d 操作
                    set_feature_mask_for_myconv2d(module, feature_mask_values)
        elif args.model == 'preactresnet':
            for name, module in model.named_children():
                if isinstance(module, MaskConv2d):
                    set_feature_mask_for_myconv2d(module, feature_mask_values)
                elif isinstance(module, nn.Sequential):
                    for name, sub_module in module.named_children():
                        premodule = sub_module.named_children()
                        for name, pre_module in premodule:
                            if isinstance(pre_module, MaskConv2d):
                                set_feature_mask_for_myconv2d(pre_module, feature_mask_values)
                            elif isinstance(pre_module, nn.Sequential):
                                for name, pre_sub_module in pre_module.named_children():
                                    if isinstance(pre_sub_module, MaskConv2d):
                                        set_feature_mask_for_myconv2d(pre_sub_module, feature_mask_values)

    # _, _, init_sparsity = PA.cal_sparsity(model)
    # switch to train mode
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

        # update coverage rate
        if args.prune and epoch >= args.first_epoch:
            # 设置初始覆盖率和最终覆盖率
            change_ratio = (args.final_cov_rate - args.initial_cov_rate)/(args.target_epoch - 10 - args.first_epoch)
            current_cov_rate = (epoch-args.first_epoch) * change_ratio + args.initial_cov_rate
            if epoch > args.target_epoch - 10:
                current_cov_rate = args.final_cov_rate
            # if epoch > args.target_epoch / 4 * 3:
            #     current_cov_rate = args.final_cov_rate
            # current_cov_rate = args.final_cov_rate if epoch > args.target_epoch / 2 else args.final_cov_rate - args.final_cov_rate * (
            #         1 - epoch / args.target_epoch) ** 3

            if globals()['iterations'] % args.prune_freq == 0:
                threshold = PA.get_weight_threshold(model, current_cov_rate, args)
                PA.weight_prune(model, threshold, args)

        if epoch < args.first_epoch:
            output = model(input, 0, 0)
            output_full = model(input, 1, 0)
        else:
            output = model(input, 0, 1)
            output_full = model(input, 1, 1)

        if epoch < args.first_epoch:
            loss = criterion(output, target)
        elif epoch < args.warmup_loss:
            loss = criterion(output, target) + criterion(output_full, target)
        else:
            loss = criterion(output, target) + criterion(output_full, target) + criterion_kl(output,
                                                                                             output_full) + criterion_kl(
                output_full, output)
        # else:
        #     loss = criterion(output, target) + criterion(output_full, target)

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

        # if i % args.print_freq == 0:
        if globals()['iterations'] % args.prune_freq == 0:
            progress.print(i)

        end = time.time()

        # end of one mini-batch
        globals()['iterations'] += 1

    print('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def validate(args, val_loader, epoch, model, criterion):
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

    return top1.avg, top5.avg


def validate_t(args, val_loader, epoch, model, criterion):
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

    return top1.avg, top5.avg


if __name__ == '__main__':
    start_time = time.time()
    args = hyperparam()
    print(args)
    main(args)
    elapsed_time = time.time() - start_time
    print('====> total time: {}h {}m {:.2f}s'.format(
        int(elapsed_time // 3600), int((elapsed_time % 3600) // 60), elapsed_time % 60))