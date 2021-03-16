def get_mean(norm_value=255, model='I3D'):
    assert model in ['I3D', 'resnet_50', 'slowfast', 'I3D_BSL', 'I3D_flow', 'sgn_pose', 'msg3d_pose', 'I3D_depth',
                     'resnet_50_part', 'I3D_BSL_part', 'I3D_BSL_face', 'I3D_BSL_lhand', 'I3D_BSL_rhand']
    if model == 'I3D':
        return [
            0.5, 0.5, 0.5
        ]
    elif model == 'resnet_50':
        return [
            114.75 / norm_value, 114.75 / norm_value,
            114.75 / norm_value
        ]
    elif model == 'slowfast':
        return [
            114.75 / norm_value, 114.75 / norm_value,
            114.75 / norm_value
        ]
    elif model == 'I3D_BSL':
        return [
            0.5, 0.5, 0.5
        ]
    elif model == 'I3D_flow':
        return [
            0.5, 0.5, 0.5
        ]
    elif model == 'I3D_depth':
        return [
            0.5, 0.5, 0.5
        ]
    elif model == 'resnet_50_part':
        return [
            114.75 / norm_value, 114.75 / norm_value,
            114.75 / norm_value
        ]
    elif model == 'I3D_BSL_part':
        return [
            0.5, 0.5, 0.5
        ]

def get_std(norm_value=255, model='None'):
    if model == 'I3D':
        return [0.5] * 3
    elif model == 'resnet_50':
        return [
            57.375 / norm_value, 57.375 / norm_value,
            57.375 / norm_value
        ]
    elif model == 'slowfast':
        return [
            57.375 / norm_value, 57.375 / norm_value,
            57.375 / norm_value
        ]
    elif model == 'I3D_BSL':
        return [0.5] * 3
    elif model == 'I3D_flow':
        return [0.5] * 3
    elif model == 'I3D_depth':
        return [0.5] * 3
    elif model == 'resnet_50_part':
        return [
            57.375 / norm_value, 57.375 / norm_value,
            57.375 / norm_value
        ]
    elif model == 'I3D_BSL_part':
        return [0.5] * 3