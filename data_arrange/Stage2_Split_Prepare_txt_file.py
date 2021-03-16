import os
import copy

data_root = '/data3/alexhu/Datasets/AUTSL/label/'

with open(os.path.join(data_root, 'train_labels.csv'), 'r') as f:
    total_train = []
    total_class = []
    total_signer = []
    total_line = []
    total_sample = []
    class_num = [0] * 226
    signer_num = [0] * 43
    signer_performed_class = []
    for i in range(43):
        signer_performed_class.append([])

    for line in f.readlines():
        line = line.strip()
        name, label = line.split(',')
        signer_id = int(name.split('_')[0][6:])
        samples = int(name.split('_')[1][6:])
        label = int(label)
        total_line.append(line)
        total_train.append(name)
        if label not in total_class:
            total_class.append(label)
        if signer_id not in total_signer:
            total_signer.append(signer_id)
        total_sample.append(samples)

        # if signer_id >= 40:
        #     continue
        signer_num[signer_id] += 1
        class_num[label] += 1
        if label not in signer_performed_class[signer_id]:
            signer_performed_class[signer_id].append(label)

print('TRAIN SET SUMMARY')
print(len(total_train), len(total_class), len(total_signer))
print('train set signer id', total_signer)
print('per signer videos:', signer_num)
print('per class num, max & min', max(class_num), min(class_num))
print('signer performed class num')
print([len(x) for x in signer_performed_class])

with open('Split/classInd.txt', 'w') as f:
    total_class = sorted(total_class)
    for i in range(len(total_class)):
        line = str(total_class[i]) + ' ' + str(total_class[i]) + '\n'
        f.writelines(line)

with open('Split/trainlist01.txt', 'w') as f:
    for i in range(len(total_line)):
        name, label = total_line[i].split(',')
        signer_id = int(name.split('_')[0][6:])
        if signer_id >= 40:
            continue
        name = str(label) + '/' + name + '.mp4'
        cur_line = name + ' ' + label + '\n'
        f.writelines(cur_line)

with open('Split/testlist01.txt', 'w') as f:
    for i in range(len(total_line)):
        name, label = total_line[i].split(',')
        signer_id = int(name.split('_')[0][6:])
        if signer_id < 40:
            continue
        name = str(label) + '/' + name + '.mp4'
        cur_line = name + '\n'
        f.writelines(cur_line)


val_path ='/data3/alexhu/Datasets/AUTSL/Archive/raw/val/'
video_list = os.listdir(val_path)
val_total_signer = []
for i in range(len(video_list)):
    name = video_list[i]
    signer_id = int(name.split('_')[0][6:])
    samples = int(name.split('_')[1][6:])
    if signer_id not in val_total_signer:
        val_total_signer.append(signer_id)
print('val signer num:', len(val_total_signer), 'val signer id', val_total_signer)
print('val samples', len(video_list))



