import torch
import time
import sys
from utils import AverageMeter, calculate_accuracy


def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets, bbox) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        with torch.no_grad():
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            if opt.model == 'slowfast':
                slow = inputs[:, :, ::8, :, :]
                fast = inputs[:, :, ::2, :, :]
                outputs = model([slow, fast])
            else:
                outputs = model(inputs)


        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}] '
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
              'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
              'Loss {loss.val:.4f} ({loss.avg:.4f}) '
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies))

    logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

    return losses.avg
