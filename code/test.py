import torch
import torch.nn.functional as F
import time
import json
import metric
import numpy as np
import os

def calculate_video_results(output_buffer, video_id, test_results, class_names):
    video_outputs = torch.stack(output_buffer)
    average_scores = torch.mean(video_outputs, dim=0)
    sorted_scores, locs = torch.topk(average_scores, k=10)

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[locs[i].item()],
            'score': sorted_scores[i].item()
        })

    test_results['results'][video_id] = video_results


def test(data_loader, model, opt, class_names):
    print('test')

    model.eval()

    # eval metrics
    metrics = metric.MetricList(
                                metric.Accuracy(topk=1, name="top1"),
                                metric.Accuracy(topk=2, name="top2"),
                                metric.Accuracy(topk=3, name="top3"),
                                metric.Accuracy(topk=4, name="top4"),
                                metric.Accuracy(topk=5, name="top5"),)
    metrics.reset()


    avg_score = {}
    sum_batch_elapse = 0.
    sum_batch_inst = 0
    duplication = 1
    total_round = 1

    out_target = []
    out_output = []

    with open('datasets/template.csv', 'r') as f:
        template_sample = {}
        for line in f.readlines():
            name = line.split(',')[0]
            template_sample[name] = -1

    interval = data_loader.__len__() // 10
    for i_round in range(total_round):

        i_batch = 0
        print("round #{}/{}".format(i_round, total_round))

        with torch.no_grad():
            for i, (inputs, targets, bbox) in enumerate(data_loader):
                # data_time.update(time.time() - end_time)
                batch_start_time = time.time()
                targets_ori = targets[0].cuda()

                if opt.model == 'slowfast':
                    slow = inputs[:, :, ::8, :, :]
                    fast = inputs[:, :, ::2, :, :]
                    outputs = model([slow, fast])
                else:
                    outputs = model(inputs)

                output_np = outputs.data.cpu().numpy()
                target_np = targets_ori.data.cpu().numpy()
                out_output.append(output_np)
                out_target.append(target_np[:, np.newaxis])


                sum_batch_elapse += time.time() - batch_start_time
                sum_batch_inst += 1
                if not opt.no_softmax_in_test:
                    outputs = F.softmax(outputs, dim=1)

                outputs = outputs.data.cpu()
                # targets = targets.cpu()


                for i_item in range(0, outputs.shape[0]):
                    output_i = outputs[i_item, :].view(1, -1)
                    target_i = torch.LongTensor([targets[0][i_item]])
                    video_subpath_i = targets[1][i_item]
                    if video_subpath_i in avg_score:
                        avg_score[video_subpath_i][1] += output_i
                        avg_score[video_subpath_i][2] += 1
                        duplication = 0.92 * duplication + 0.08 * avg_score[video_subpath_i][2]
                    else:
                        avg_score[video_subpath_i] = [torch.LongTensor(target_i.numpy().copy()),
                                                      torch.FloatTensor(output_i.numpy().copy()),
                                                      1]  # the last one is counter

                    # show progress
                if (i_batch % interval) == 0:
                    metrics.reset()
                    for _, video_info in avg_score.items():
                        target, pred, _ = video_info
                        metrics.update([pred], target)
                    name_value = metrics.get_name_value()
                    print(
                        "{:.1f}%, {:.1f} \t| Batch [0,{}]    \tAvg: {} = {:.5f}, {} = {:.5f}".format(
                            float(100 * i_batch) / data_loader.__len__(), \
                            duplication, \
                            i_batch, \
                            name_value[0][0][0], name_value[0][0][1], \
                            name_value[1][0][0], name_value[1][0][1]))
                i_batch += 1

        # finished
        print("Evaluation one epoch Finished!")

        # savefig
        output_array = np.concatenate(out_output, axis=0)
        target_array = np.concatenate(out_target, axis=0)
        if opt.annotation_path.endswith('split.json'):
            name = 'AUTSL_' + opt.model + '.npy'
            pkl_name = 'AUTSL_' + opt.model + '2_all.pkl'
        else:
            name = 'AUTSL_' + opt.model + '_all.npy'
            pkl_name = 'AUTSL_' + opt.model + '_all.pkl'
        # np.save(os.path.join(name), output_array, allow_pickle=False)

        import pickle
        with open(pkl_name, 'wb') as f:
            pickle.dump(avg_score, f)


        metrics.reset()
        class_num = {}
        class_acc = {}
        for _, video_info in avg_score.items():
            # total video
            target, pred, _ = video_info
            metrics.update([pred], target)

            # class acc
            if target.item() not in class_num:
                class_num[target.item()] = 1
            else:
                class_num[target.item()] += 1

            _, pred_topk = pred.topk(1, 1, True, True)

            pred_topk = pred_topk.t()
            correct = pred_topk.eq(target.view(1, -1).expand_as(pred_topk))
            if target.item() not in class_acc:
                # class_acc[target.item()] = correct.item()
                class_acc[target.item()] = float(correct.view(-1).float().sum(0, keepdim=True).numpy())
            else:
                # class_acc[target.item()] += correct.item()
                class_acc[target.item()] += float(correct.view(-1).float().sum(0, keepdim=True).numpy())



        for video_name, video_info in avg_score.items():
            target, pred, _ = video_info
            template_sample[video_name] = torch.argmax(pred).item()
        # with open('predictions.csv', 'w') as f2:
        #     for k, v in template_sample.items():
        #         line = k + ',' + str(v) + '\n'
        #         f2.writelines(line)


        print("Total time cost: {:.1f} sec".format(sum_batch_elapse))
        print("Speed: {:.4f} samples/sec".format(
            opt.batch_size * sum_batch_inst / sum_batch_elapse))
        print("Accuracy:")
        print(json.dumps(metrics.get_name_value(), indent=4, sort_keys=True))