import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from spatial_transforms_flow import (
    Compose_Flow, Normalize_Flow, Scale_Flow, CenterCrop_Flow, CornerCrop_Flow, MultiScaleCornerCrop_Flow,
    MultiScaleRandomCrop_Flow, RandomHorizontalFlip_Flow, ToTensor_Flow)
from temporal_transforms import LoopPadding, TemporalRandomCrop, TemporalCenterCrop, TemporalBeginCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger
from train import train_epoch
from validation import val_epoch
import test
import os
from torch.utils.data import DataLoader
import random


def setup_seed(seed=8):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = opt.model
    opt.mean = get_mean(opt.norm_value, model=opt.model)
    opt.std = get_std(opt.norm_value, model=opt.model)
    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    setup_seed(opt.manual_seed)
    model, parameters = generate_model(opt)
    print(model)
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if not opt.no_train:

        ##--------------------------------------------------------------------------------------------
        if opt.model == 'I3D':
            assert opt.train_crop in ['random', 'corner', 'center']
            if opt.train_crop == 'random':
                crop_method = MultiScaleRandomCrop([0.875], opt.sample_size)
            elif opt.train_crop == 'corner':
                crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
            elif opt.train_crop == 'center':
                crop_method = MultiScaleCornerCrop(
                    opt.scales, opt.sample_size, crop_positions=['c'])
            spatial_transform = Compose([
                Scale((256, 256)),
                crop_method,
                RandomHorizontalFlip(),
                ToTensor(opt.norm_value), norm_method
            ])
            temporal_transform = TemporalRandomCrop(opt.sample_duration, 1)
        elif opt.model == 'resnet_50':
            assert opt.train_crop in ['random', 'corner', 'center']
            if opt.train_crop == 'random':
                crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)  # [1, 0.9, 0.875]
            elif opt.train_crop == 'corner':
                crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
            elif opt.train_crop == 'center':
                crop_method = MultiScaleCornerCrop(
                    opt.scales, opt.sample_size, crop_positions=['c'])
            spatial_transform = Compose([
                crop_method,
                RandomHorizontalFlip(),
                ToTensor(opt.norm_value), norm_method
            ])
            temporal_transform = TemporalRandomCrop(opt.sample_duration, 1)
        elif opt.model == 'slowfast':
            assert opt.train_crop in ['random', 'corner', 'center']
            if opt.train_crop == 'random':
                crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
            elif opt.train_crop == 'corner':
                crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
            elif opt.train_crop == 'center':
                crop_method = MultiScaleCornerCrop(
                    opt.scales, opt.sample_size, crop_positions=['c'])
            spatial_transform = Compose([
                crop_method,
                RandomHorizontalFlip(),
                ToTensor(opt.norm_value), norm_method
            ])
            temporal_transform = TemporalRandomCrop(opt.sample_duration, 1)
        elif opt.model == 'I3D_BSL':
            assert opt.train_crop in ['random', 'corner', 'center']
            if opt.train_crop == 'random':
                crop_method = MultiScaleRandomCrop([0.875], opt.sample_size)
            elif opt.train_crop == 'corner':
                crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
            elif opt.train_crop == 'center':
                crop_method = MultiScaleCornerCrop(
                    opt.scales, opt.sample_size, crop_positions=['c'])
            spatial_transform = Compose([
                Scale((256, 256)),
                crop_method,
                RandomHorizontalFlip(),
                ToTensor(opt.norm_value), norm_method
            ])
            temporal_transform = TemporalRandomCrop(opt.sample_duration, 1)
        elif opt.model == 'I3D_flow':
            assert opt.train_crop in ['random', 'corner', 'center']
            if opt.train_crop == 'random':
                crop_method = MultiScaleRandomCrop_Flow([0.875], opt.sample_size)
            elif opt.train_crop == 'corner':
                crop_method = MultiScaleCornerCrop_Flow(opt.scales, opt.sample_size)
            elif opt.train_crop == 'center':
                crop_method = MultiScaleCornerCrop_Flow(
                    opt.scales, opt.sample_size, crop_positions=['c'])
            spatial_transform = Compose([
                Scale_Flow((256, 256)),
                crop_method,
                RandomHorizontalFlip_Flow(),
                ToTensor_Flow(opt.norm_value), norm_method
            ])
            temporal_transform = TemporalRandomCrop(opt.sample_duration, 1)
        elif opt.model == 'sgn_pose':
            spatial_transform = None
            temporal_transform = TemporalRandomCrop(opt.sample_duration, 1)
        elif opt.model == 'msg3d_pose':
            spatial_transform = None
            temporal_transform = TemporalRandomCrop(opt.sample_duration, 1)
        elif opt.model == 'I3D_depth':
            assert opt.train_crop in ['random', 'corner', 'center']
            if opt.train_crop == 'random':
                crop_method = MultiScaleRandomCrop([0.875], opt.sample_size)
            elif opt.train_crop == 'corner':
                crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
            elif opt.train_crop == 'center':
                crop_method = MultiScaleCornerCrop(
                    opt.scales, opt.sample_size, crop_positions=['c'])
            spatial_transform = Compose([
                Scale((256, 256)),
                crop_method,
                RandomHorizontalFlip(),
                ToTensor(opt.norm_value), norm_method
            ])
            temporal_transform = TemporalRandomCrop(opt.sample_duration, 1)
        elif opt.model == 'resnet_50_part':
            assert opt.train_crop in ['random', 'corner', 'center']
            if opt.train_crop == 'random':
                crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)  # [1, 0.9, 0.875]
            elif opt.train_crop == 'corner':
                crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
            elif opt.train_crop == 'center':
                crop_method = MultiScaleCornerCrop(
                    opt.scales, opt.sample_size, crop_positions=['c'])
            spatial_transform = Compose([
                crop_method,
                RandomHorizontalFlip(),
                ToTensor(opt.norm_value), norm_method
            ])
            temporal_transform = TemporalRandomCrop(opt.sample_duration, 1)
        elif opt.model in ['I3D_BSL_part', 'I3D_BSL_face', 'I3D_BSL_lhand', 'I3D_BSL_rhand']:
            assert opt.train_crop in ['random', 'corner', 'center']
            if opt.train_crop == 'random':
                crop_method = MultiScaleRandomCrop([0.875], opt.sample_size)
            elif opt.train_crop == 'corner':
                crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
            elif opt.train_crop == 'center':
                crop_method = MultiScaleCornerCrop(
                    opt.scales, opt.sample_size, crop_positions=['c'])
            spatial_transform = Compose([
                Scale((256, 256)),
                crop_method,
                RandomHorizontalFlip(),
                ToTensor(opt.norm_value), norm_method
            ])
            temporal_transform = TemporalRandomCrop(opt.sample_duration, 1)


        target_transform = ClassLabel()
        if opt.model.endswith('flow'):
            training_data = get_training_set(opt, spatial_transform,
                                             temporal_transform, target_transform, modality='flow')
        elif opt.model.endswith('pose'):
            training_data = get_training_set(opt, spatial_transform,
                                             temporal_transform, target_transform, modality='pose')
        elif opt.model.endswith('depth'):
            training_data = get_training_set(opt, spatial_transform,
                                             temporal_transform, target_transform, modality='depth')
        elif opt.model.endswith('part'):
            training_data = get_training_set(opt, spatial_transform,
                                             temporal_transform, target_transform, modality='part')
        else:
            training_data = get_training_set(opt, spatial_transform,
                                             temporal_transform, target_transform, modality='rgb')

        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening


        ##--------------------------------------------------------------------------------------------
        if opt.model == 'I3D':
            optimizer = optim.SGD(
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                # dampening=dampening,
                weight_decay=opt.weight_decay,
                nesterov=True)
            scheduler = lr_scheduler.MultiStepLR(optimizer, [25, 40, 45, 50, 55, 60], gamma=0.1)
        elif opt.model == 'resnet_50':
            optimizer = optim.SGD(
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                # dampening=dampening,
                weight_decay=opt.weight_decay,
                nesterov=True)
            scheduler = lr_scheduler.MultiStepLR(optimizer, [25, 40, 50], gamma=0.1)
        elif opt.model == 'slowfast':
            optimizer = optim.SGD(
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                # dampening=dampening,
                weight_decay=opt.weight_decay,
                nesterov=True)
            scheduler = lr_scheduler.MultiStepLR(optimizer, [25, 40, 45, 50, 55, 60], gamma=0.1)
        elif opt.model == 'I3D_BSL':
            optimizer = optim.SGD(
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                # dampening=dampening,
                weight_decay=opt.weight_decay,
                nesterov=True)
            scheduler = lr_scheduler.MultiStepLR(optimizer, [15, 25, 40, 45, 50, 55, 60], gamma=0.1)
        elif opt.model == 'I3D_flow':
            optimizer = optim.SGD(
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                # dampening=dampening,
                weight_decay=opt.weight_decay,
                nesterov=True)
            scheduler = lr_scheduler.MultiStepLR(optimizer, [25, 40, 45, 50, 55, 60], gamma=0.1)
        elif opt.model == 'sgn_pose':
            optimizer = optim.Adam(
                parameters,
                lr=opt.learning_rate,
                weight_decay=opt.weight_decay)
            scheduler = lr_scheduler.MultiStepLR(optimizer, [60, 80, 100], gamma=0.1)
        elif opt.model == 'msg3d_pose':
            optimizer = optim.SGD(
                parameters,
                lr=opt.learning_rate,
                momentum=0.9,
                nesterov=True,
                weight_decay=opt.weight_decay)
            scheduler = lr_scheduler.MultiStepLR(optimizer, [10, 20, 30, 40, 50], gamma=0.1)
        elif opt.model == 'I3D_depth':
            optimizer = optim.SGD(
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                # dampening=dampening,
                weight_decay=opt.weight_decay,
                nesterov=True)
            scheduler = lr_scheduler.MultiStepLR(optimizer, [25, 40, 45, 50, 55, 60], gamma=0.1)
        elif opt.model == 'resnet_50_part':
            optimizer = optim.SGD(
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                # dampening=dampening,
                weight_decay=opt.weight_decay,
                nesterov=True)
            scheduler = lr_scheduler.MultiStepLR(optimizer, [25, 40, 45, 50, 55, 60], gamma=0.1)
        elif opt.model in ['I3D_BSL_part', 'I3D_BSL_face', 'I3D_BSL_lhand', 'I3D_BSL_rhand']:
            optimizer = optim.SGD(
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                # dampening=dampening,
                weight_decay=opt.weight_decay,
                nesterov=True)
            scheduler = lr_scheduler.MultiStepLR(optimizer, [15, 25, 40, 45, 50, 55, 60], gamma=0.1)

    if not opt.no_val:

        ##--------------------------------------------------------------------------------------------
        if opt.model == 'I3D':
            spatial_transform = Compose([
                Scale((256, 256)),
                CenterCrop(224),
                ToTensor(opt.norm_value), norm_method
            ])
            temporal_transform = LoopPadding(0)
            target_transform = ClassLabel()
            validation_data = get_validation_set(
                opt, spatial_transform, temporal_transform, target_transform)
            val_loader = torch.utils.data.DataLoader(
                validation_data,
                batch_size=1,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
        elif opt.model == 'resnet_50':
            spatial_transform = Compose([
                Scale(256),
                CenterCrop(256),
                ToTensor(opt.norm_value), norm_method
            ])
            temporal_transform = TemporalCenterCrop(opt.sample_duration, 1)
            target_transform = ClassLabel()
            validation_data = get_validation_set(
                opt, spatial_transform, temporal_transform, target_transform)
            val_loader = torch.utils.data.DataLoader(
                validation_data,
                batch_size=16,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
        elif opt.model == 'slowfast':
            spatial_transform = Compose([
                Scale(256),
                CenterCrop(256),
                ToTensor(opt.norm_value), norm_method
            ])
            temporal_transform = TemporalCenterCrop(opt.sample_duration, 1)
            target_transform = ClassLabel()
            validation_data = get_validation_set(
                opt, spatial_transform, temporal_transform, target_transform)
            val_loader = torch.utils.data.DataLoader(
                validation_data,
                batch_size=16,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
        elif opt.model == 'I3D_BSL':
            spatial_transform = Compose([
                Scale((256, 256)),
                CenterCrop(224),
                ToTensor(opt.norm_value), norm_method
            ])
            temporal_transform = LoopPadding(0)
            target_transform = ClassLabel()
            validation_data = get_validation_set(
                opt, spatial_transform, temporal_transform, target_transform)
            val_loader = torch.utils.data.DataLoader(
                validation_data,
                batch_size=1,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
        elif opt.model == 'I3D_flow':
            spatial_transform = Compose([
                Scale_Flow((256, 256)),
                CenterCrop_Flow(224),
                ToTensor_Flow(opt.norm_value), norm_method
            ])
            temporal_transform = LoopPadding(0)
            target_transform = ClassLabel()
            validation_data = get_validation_set(
                opt, spatial_transform, temporal_transform, target_transform, modality='flow')
            val_loader = torch.utils.data.DataLoader(
                validation_data,
                batch_size=1,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
        elif opt.model == 'sgn_pose':
            spatial_transform = None
            temporal_transform = TemporalCenterCrop(opt.sample_duration, 1)
            target_transform = ClassLabel()
            validation_data = get_validation_set(
                opt, spatial_transform, temporal_transform, target_transform, modality='pose')
            val_loader = torch.utils.data.DataLoader(
                validation_data,
                batch_size=64,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
        elif opt.model == 'msg3d_pose':
            spatial_transform = None
            temporal_transform = TemporalCenterCrop(opt.sample_duration, 1)
            target_transform = ClassLabel()
            validation_data = get_validation_set(
                opt, spatial_transform, temporal_transform, target_transform, modality='pose')
            val_loader = torch.utils.data.DataLoader(
                validation_data,
                batch_size=64,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
        elif opt.model == 'I3D_depth':
            spatial_transform = Compose([
                Scale((256, 256)),
                CenterCrop(224),
                ToTensor(opt.norm_value), norm_method
            ])
            temporal_transform = LoopPadding(0)
            target_transform = ClassLabel()
            validation_data = get_validation_set(
                opt, spatial_transform, temporal_transform, target_transform, modality='depth')
            val_loader = torch.utils.data.DataLoader(
                validation_data,
                batch_size=1,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
        elif opt.model == 'resnet_50_part':
            spatial_transform = Compose([
                Scale(256),
                CenterCrop(256),
                ToTensor(opt.norm_value), norm_method
            ])
            temporal_transform = TemporalCenterCrop(opt.sample_duration, 1)
            target_transform = ClassLabel()
            validation_data = get_validation_set(
                opt, spatial_transform, temporal_transform, target_transform, modality='part')
            val_loader = torch.utils.data.DataLoader(
                validation_data,
                batch_size=16,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
        elif opt.model in ['I3D_BSL_part', 'I3D_BSL_face', 'I3D_BSL_lhand', 'I3D_BSL_rhand']:
            spatial_transform = Compose([
                Scale((256, 256)),
                CenterCrop(224),
                ToTensor(opt.norm_value), norm_method
            ])
            temporal_transform = LoopPadding(0)
            target_transform = ClassLabel()
            validation_data = get_validation_set(
                opt, spatial_transform, temporal_transform, target_transform, modality='part')
            val_loader = torch.utils.data.DataLoader(
                validation_data,
                batch_size=1,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)

        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

        for i in range(opt.begin_epoch-1):
            scheduler.step()

    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)
        if not opt.no_val:
            validation_loss = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger)


        if not opt.no_train and not opt.no_val:
            # scheduler.step(validation_loss)
            scheduler.step()

    if opt.test:

        ##--------------------------------------------------------------------------------------------
        if opt.model == 'I3D':
            spatial_transform = Compose([
                Scale((256, 256)),
                CenterCrop(224),
                ToTensor(opt.norm_value), norm_method
            ])
            temporal_transform = LoopPadding(64)
        elif opt.model == 'resnet_50':
            spatial_transform = Compose([
                Scale(256),
                # CornerCrop(opt.sample_size, opt.crop_position_in_test),
                CenterCrop(256),
                ToTensor(opt.norm_value), norm_method
            ])
            temporal_transform = TemporalCenterCrop(opt.sample_duration, 1)
        elif opt.model == 'slowfast':
            spatial_transform = Compose([
                Scale(256),
                # CornerCrop(opt.sample_size, opt.crop_position_in_test),
                CenterCrop(256),
                ToTensor(opt.norm_value), norm_method
            ])
            temporal_transform = TemporalCenterCrop(opt.sample_duration, 1)
        elif opt.model == 'I3D_BSL':
            spatial_transform = Compose([
                Scale((256, 256)),
                CenterCrop(224),
                ToTensor(opt.norm_value), norm_method
            ])
            temporal_transform = LoopPadding(64)
        elif opt.model == 'I3D_flow':
            spatial_transform = Compose([
                Scale_Flow((256, 256)),
                CenterCrop_Flow(224),
                ToTensor_Flow(opt.norm_value), norm_method
            ])
            temporal_transform = LoopPadding(64)
        elif opt.model == 'sgn_pose':
            spatial_transform = None
            temporal_transform = TemporalCenterCrop(opt.sample_duration, 1)
        elif opt.model == 'msg3d_pose':
            spatial_transform = None
            temporal_transform = TemporalCenterCrop(opt.sample_duration, 1)
        elif opt.model == 'I3D_depth':
            spatial_transform = Compose([
                Scale((256, 256)),
                CenterCrop(224),
                ToTensor(opt.norm_value), norm_method
            ])
            temporal_transform = LoopPadding(64)
        elif opt.model == 'resnet_50_part':
            spatial_transform = Compose([
                Scale(256),
                # CornerCrop(opt.sample_size, opt.crop_position_in_test),
                CenterCrop(256),
                ToTensor(opt.norm_value), norm_method
            ])
            temporal_transform = TemporalCenterCrop(opt.sample_duration, 1)
        elif opt.model in ['I3D_BSL_part', 'I3D_BSL_face', 'I3D_BSL_lhand', 'I3D_BSL_rhand']:
            spatial_transform = Compose([
                Scale((256, 256)),
                CenterCrop(224),
                ToTensor(opt.norm_value), norm_method
            ])
            temporal_transform = LoopPadding(64)

        target_transform = VideoID()
        target_transform = TargetCompose([ClassLabel(), VideoID()])

        if opt.model.endswith('flow'):
            test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                     target_transform, modality='flow')
        elif opt.model.endswith('pose'):
            test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                     target_transform, modality='pose')
        elif opt.model.endswith('depth'):
            test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                     target_transform, modality='depth')
        elif opt.model.endswith('part'):
            test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                     target_transform, modality='part')
        else:
            test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                     target_transform)
        print(len(test_data))

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        test.test(test_loader, model, opt, test_data.class_names)
