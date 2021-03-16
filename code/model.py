import torch
from torch import nn
from models import resnet_50, i3dpt, slowfast_model, i3d_bsl, SGN
from models.msg3d import msg3d


def generate_model(opt):
    assert opt.model in [
        'resnet_50', 'I3D', 'slowfast', 'I3D_BSL', 'I3D_flow', 'sgn_pose', 'msg3d_pose', 'I3D_depth', 'resnet_50_part', 'I3D_BSL_part', 'I3D_BSL_face', 'I3D_BSL_lhand', 'I3D_BSL_rhand'
    ]

    if opt.model == 'resnet_50':
        from models.resnet_50 import get_fine_tuning_parameters
        model = resnet_50.i3_res50(num_classes=400)
    elif opt.model == 'resnet_50_nl':
        from models.resnet_50 import get_fine_tuning_parameters
        model = resnet_50.i3_res50_nl(num_classes=400)
    elif opt.model == 'I3D':
        model = i3dpt.I3D(num_classes=400)
    elif opt.model == 'slowfast':
        model = slowfast_model.model_build()
    elif opt.model == 'I3D_BSL':
        model = i3d_bsl.InceptionI3d(num_classes=opt.n_classes)
    elif opt.model == 'I3D_flow':
        model = i3dpt.I3D(num_classes=400, modality='flow')
    elif opt.model == 'sgn_pose':
        model = SGN.SGN(opt.n_classes, opt.sample_duration)
    elif opt.model == 'msg3d_pose':
        model = msg3d.Model(
            num_class=opt.n_classes,
            num_point=65,
            num_person=1,
            num_gcn_scales=8,
            num_g3d_scales=8,
            graph='graph.AUTSL.AdjMatrixGraph'
        )
    elif opt.model == 'I3D_depth':
        model = i3dpt.I3D(num_classes=400, modality='depth')
    elif opt.model == 'resnet_50_part':
        from models.resnet_50 import get_fine_tuning_parameters
        model = resnet_50.i3_res50(num_classes=400)
    elif opt.model in ['I3D_BSL_part', 'I3D_BSL_face', 'I3D_BSL_lhand', 'I3D_BSL_rhand']:
        model = i3d_bsl.InceptionI3d(num_classes=opt.n_classes)

    if not opt.no_cuda:
        model = model.cuda()

        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)

            if opt.no_train and opt.no_val:
                if opt.model == 'I3D':
                    model.replace_logits(opt.n_finetune_classes)
                    model = nn.DataParallel(model, device_ids=None)
                    model.load_state_dict(pretrain['state_dict'])
                elif opt.model == 'resnet_50':
                    model = nn.DataParallel(model, device_ids=None)
                    model.module.fc = nn.Linear(model.module.fc.in_features,
                                                opt.n_finetune_classes)
                    model.module.fc = model.module.fc.cuda()
                    model.load_state_dict(pretrain['state_dict'])
                elif opt.model == 'slowfast':
                    model = nn.DataParallel(model, device_ids=None)
                    model.module.head.projection = nn.Linear(2304, opt.n_finetune_classes)
                    model.module.head.projection = model.module.head.projection.cuda()
                    model.load_state_dict(pretrain['state_dict'])
                elif opt.model == 'I3D_BSL':
                    model = nn.DataParallel(model, device_ids=None)
                    model.load_state_dict(pretrain['state_dict'])
                elif opt.model == 'I3D_flow':
                    model.replace_logits(opt.n_finetune_classes)
                    model = nn.DataParallel(model, device_ids=None)
                    model.load_state_dict(pretrain['state_dict'])
                elif opt.model == 'sgn_pose':
                    model = nn.DataParallel(model, device_ids=None)
                    model.load_state_dict(pretrain['state_dict'])
                elif opt.model == 'msg3d_pose':
                    model = nn.DataParallel(model, device_ids=None)
                    model.load_state_dict(pretrain['state_dict'])
                elif opt.model == 'I3D_depth':
                    model.replace_logits(opt.n_finetune_classes)
                    model = nn.DataParallel(model, device_ids=None)
                    model.load_state_dict(pretrain['state_dict'])
                elif opt.model == 'resnet_50_part':
                    model = nn.DataParallel(model, device_ids=None)
                    model.module.fc = nn.Linear(model.module.fc.in_features,
                                                opt.n_finetune_classes)
                    model.module.fc = model.module.fc.cuda()
                    model.load_state_dict(pretrain['state_dict'])
                elif opt.model in ['I3D_BSL_part', 'I3D_BSL_face', 'I3D_BSL_lhand', 'I3D_BSL_rhand']:
                    model = nn.DataParallel(model, device_ids=None)
                    model.load_state_dict(pretrain['state_dict'])

            else:
                if opt.model == 'I3D':
                    model.load_state_dict(pretrain)
                    model.replace_logits(opt.n_finetune_classes)
                    model = nn.DataParallel(model, device_ids=None)
                    return model, model.parameters()
                elif opt.model == 'resnet_50':
                    model.load_state_dict(pretrain)
                    model = nn.DataParallel(model, device_ids=None)
                    model.module.fc = nn.Linear(model.module.fc.in_features,
                                                opt.n_finetune_classes)
                    model.module.fc = model.module.fc.cuda()
                    # parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
                    return model, model.parameters()
                elif opt.model == 'slowfast':
                    model = nn.DataParallel(model, device_ids=None)
                    model.module.head.projection = nn.Linear(2304, opt.n_finetune_classes)
                    model.module.head.projection = model.module.head.projection.cuda()
                    return model, model.parameters()
                elif opt.model == 'I3D_BSL':
                    model = nn.DataParallel(model, device_ids=None)
                    model.load_state_dict(pretrain['state_dict'])
                    model.module.replace_logits(opt.n_finetune_classes)
                    model = model.cuda()
                    return model, model.parameters()
                elif opt.model == 'I3D_flow':
                    model.load_state_dict(pretrain)
                    model.replace_logits(opt.n_finetune_classes)
                    model = nn.DataParallel(model, device_ids=None)
                    return model, model.parameters()
                elif opt.model == 'sgn_pose':
                    model = nn.DataParallel(model, device_ids=None)
                    model.load_state_dict(pretrain['state_dict'])
                    model.module.fc = nn.Linear(model.module.fc.in_features,
                                                opt.n_finetune_classes)
                    model.module.fc = model.module.fc.cuda()
                    return model, model.parameters()
                elif opt.model == 'msg3d_pose':
                    model = nn.DataParallel(model, device_ids=None)
                    # model.load_state_dict(pretrain['state_dict'])
                    model.module.fc = nn.Linear(model.module.fc.in_features,
                                                opt.n_finetune_classes)
                    model.module.fc = model.module.fc.cuda()
                    return model, model.parameters()
                elif opt.model == 'I3D_depth':
                    model.load_state_dict(pretrain)
                    model.replace_logits(opt.n_finetune_classes)
                    model = nn.DataParallel(model, device_ids=None)
                    return model, model.parameters()
                elif opt.model == 'resnet_50_part':
                    model.load_state_dict(pretrain)
                    model = nn.DataParallel(model, device_ids=None)
                    model.module.fc = nn.Linear(model.module.fc.in_features,
                                                opt.n_finetune_classes)
                    model.module.fc = model.module.fc.cuda()
                    # parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
                    return model, model.parameters()
                elif opt.model in ['I3D_BSL_part', 'I3D_BSL_face', 'I3D_BSL_lhand', 'I3D_BSL_rhand']:
                    model = nn.DataParallel(model, device_ids=None)
                    model.load_state_dict(pretrain['state_dict'])
                    model.module.replace_logits(opt.n_finetune_classes)
                    model = model.cuda()
                    return model, model.parameters()

    return model, model.parameters()
