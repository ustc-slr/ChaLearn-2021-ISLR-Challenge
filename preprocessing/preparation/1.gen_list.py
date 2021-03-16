import glob
import os


def gen_data_list(base, file, task):
    r"""generate training list from csv and verify the files
    """    
    base = os.path.join(base, task)
    with open(f'data/AUTSL/raw/{file}', 'r') as f:
        lines = f.readlines()
    lines = [x.strip().split(',') for x in lines]
    if file[-10:-4] == 'random':
        lines = [[x[0], -1] for x in lines]
    else:
        lines = [[x[0], int(x[1])] for x in lines]

    for line in lines:
        assert os.path.exists(os.path.join(base, line[0]+'_color.mp4'))
        assert os.path.exists(os.path.join(base, line[0]+'_depth.mp4'))

    with open(f'preparation/pre_data/file_{task}.txt', 'w') as f:
        lines = ['{:s},{:d}\n'.format(x[0], x[1]) for x in lines]
        f.writelines(lines)

if __name__ == "__main__":
    # gen_data_list('data/AUTSL/raw', 'train_labels.csv', 'train')
    # gen_data_list('data/AUTSL/raw', 'val_random.csv', 'val')
    gen_data_list('data/AUTSL/raw', 'test_random.csv', 'test')

