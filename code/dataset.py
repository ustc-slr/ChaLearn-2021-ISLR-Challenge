from datasets.universal import Universal

def get_training_set(opt, spatial_transform, temporal_transform,
                     target_transform, modality='rgb'):
    assert opt.dataset in ['autsl']

    if opt.dataset == 'autsl':
        training_data = Universal(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            modality=modality)

    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform, modality='rgb'):
    assert opt.dataset in ['autsl']

    if opt.dataset == 'autsl':
        validation_data = Universal(
            opt.video_path,
            opt.annotation_path,
            'testing',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration,
            modality=modality)

    return validation_data


def get_test_set(opt, spatial_transform, temporal_transform, target_transform, modality='rgb'):
    assert opt.dataset in ['autsl']
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'

    if opt.dataset == 'autsl':
        test_data = Universal(
            opt.video_path,
            opt.annotation_path,
            subset,
            1,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration,
            modality=modality)

    return test_data
