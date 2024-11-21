#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import random
import json
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import numpy as np
from configs.opts import parser
from model_src.fully_main_model import supv_main_model as main_model
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.Recorder import Recorder
from dataset.AVE_dataset import AVEDataset
# from model.FocalLoss import BinaryFocalLoss
from losses import inter_segment_sim_loss
from sklearn.metrics import average_precision_score
import pdb

# =================================  seed config ============================
SEED = 456
# SEED = 42
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# =============================================================================



def get_flag_by_gt(is_event_scores):
    # is_event_scores: [bs, 10]
    scores_pos_ind = is_event_scores #> 0.5
    pred_temp = scores_pos_ind.long() # [B, 10]
    pred = pred_temp.unsqueeze(1) # [B, 1, 10]

    pos_flag = pred.repeat(1, 10, 1) # [B, 10, 10]
    pos_flag *= pred.permute(0, 2, 1)
    neg_flag = (1 - pred).repeat(1, 10, 1) # [B, 10, 10]
    neg_flag *= pred.permute(0, 2, 1)

    return pred_temp, pos_flag, neg_flag

def main():
    # utils variable
    global args, logger, writer, dataset_configs
    # statistics variable
    global best_accuracy, best_accuracy_epoch
    best_accuracy, best_video_accuracy, best_accuracy_epoch = 0, 0, 0
    # configs
    dataset_configs = get_and_save_args(parser)
    parser.set_defaults(**dataset_configs)
    args = parser.parse_args()
    # select GPUs
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    '''Create snapshot_pred dir for copying code and saving models '''
    if not os.path.exists(args.snapshot_pref):
        os.makedirs(args.snapshot_pref)

    if os.path.isfile(args.resume):
        args.snapshot_pref = os.path.dirname(args.resume)

    logger = Prepare_logger(args, eval=args.evaluate)

    if not args.evaluate:
        logger.info(f'\nCreating folder: {args.snapshot_pref}')
        logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))
    else:
        logger.info(f'\nLog file will be save in a {args.snapshot_pref}/Eval.log.')


    '''Dataset'''
    train_dataloader = DataLoader(
        AVEDataset('./AVE_data/', split='train'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        AVEDataset('./AVE_data/', split='val'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        AVEDataset('./AVE_data/', split='test'),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    '''model setting'''
    mainModel = main_model()

    mainModel = nn.DataParallel(mainModel).cuda()
    learned_parameters = mainModel.parameters()
    optimizer = torch.optim.Adam(learned_parameters, lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[10], gamma=0.1)

    criterion = nn.BCEWithLogitsLoss().cuda()

    criterion_event = nn.CrossEntropyLoss().cuda()

    '''Resume from a checkpoint'''
    if os.path.isfile(args.resume):
        logger.info(f"\nLoading Checkpoint: {args.resume}\n")
        mainModel.load_state_dict(torch.load(args.resume))
    elif args.resume != "" and (not os.path.isfile(args.resume)):
        raise FileNotFoundError


    '''Only Evaluate'''
    if args.evaluate:
        logger.info(f"\nStart Evaluation..")
        validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch=0, eval_only=True)
        return

    # '''Tensorboard and Code backup'''
    writer = SummaryWriter(args.snapshot_pref)
    recorder = Recorder(args.snapshot_pref, ignore_folder="Exps/")
    recorder.writeopt(args)


    '''Training and Testing'''
    for epoch in range(args.n_epoch):

        logger.info(f"\tnow epoch: {epoch}")

        loss = train_epoch(mainModel, train_dataloader, criterion, criterion_event, optimizer, epoch)

        if ((epoch + 1) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
            acc, video_acc = validate_epoch(mainModel, val_dataloader, criterion, criterion_event, epoch)
            if acc > best_accuracy:
                best_accuracy = acc
                best_accuracy_epoch = epoch
                save_checkpoint(
                    mainModel.state_dict(),
                    top1=best_accuracy,
                    task='Supervised',
                    epoch=epoch + 1,
                )
            if video_acc > best_video_accuracy:
                best_video_accuracy = video_acc
        scheduler.step()

    print("-----------------------------")
    print("best acc and epoch:", best_accuracy, best_accuracy_epoch)
    print("-----------------------------")


def train_epoch(model, train_dataloader, criterion, criterion_event, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    end_time = time.time()

    model.train()

    model.double()
    optimizer.zero_grad()

    for n_iter, batch_data in enumerate(train_dataloader):

        data_time.update(time.time() - end_time)
        '''Feed input to model'''
        visual_feature, audio_feature, labels = batch_data

        labels = labels.double().cuda()
        is_event_scores, event_scores, fusion = model(visual_feature, audio_feature)
        is_event_scores = is_event_scores.transpose(1, 0).squeeze().contiguous()

        labels_foreground = labels[:, :, :-1]  # [32, 10, 28]
        labels_BCE, labels_evn = labels_foreground.max(-1)

        labels_event, _ = labels_evn.max(-1)

        loss_is_event = criterion(is_event_scores, labels_BCE.cuda())

        loss_event_class = criterion_event(event_scores, labels_event.cuda())

        fusion = fusion.transpose(1, 0)
        loss_inter = inter_segment_sim_loss(fusion, labels_BCE) * 3

        loss = loss_is_event + loss_event_class + loss_inter

        loss.backward()

        # '''Compute Accuracy'''
        # acc = compute_accuracy_supervised(is_event_scores, event_scores, labels)
        # train_acc.update(acc.item(), visual_feature.size(0) * 10)

        '''Clip Gradient'''
        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                logger.info(f'Clipping gradient: {total_norm} with coef {args.clip_gradient/total_norm}.')

        '''Update parameters'''
        optimizer.step()
        optimizer.zero_grad()

        loss_save_json.append((loss_is_event + loss_event_class + loss_inter).item())

        losses.update(loss.item(), visual_feature.size(0)*10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # '''Add loss of a iteration in Tensorboard'''
        writer.add_scalar('Train_data/loss', losses.val, epoch * len(train_dataloader) + n_iter + 1)
        #
        # '''Print logs in Terminal'''

        if n_iter % 100 == 0:
            print("loss", loss.item(), "loss_is_event", loss_is_event.item(), "loss_event_class", loss_event_class.item(),
                  "loss_inter", loss_inter.item())

            for param_group in optimizer.param_groups:
                lr_ = param_group["lr"]
                logger.info(f"\tlr: {lr_:.6f}.")

        if n_iter % args.print_freq == 0:
            logger.info(
                f'Train Epoch: [{epoch}][{n_iter}/{len(train_dataloader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
            )
        #
        '''Add loss of an epoch in Tensorboard'''
        writer.add_scalar('Train_epoch_data/epoch_loss', losses.avg, epoch)

    return losses.avg


@torch.no_grad()
def validate_epoch(model, test_dataloader, criterion, criterion_event, epoch, eval_only=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end_time = time.time()

    model.eval()
    model.double()

    for n_iter, batch_data in enumerate(test_dataloader):
        data_time.update(time.time() - end_time)

        '''Feed input to model'''
        visual_feature, audio_feature, labels = batch_data

        labels = labels.double().cuda()

        bs = visual_feature.size(0)
        is_event_scores, event_scores, _ = model(visual_feature, audio_feature)
        is_event_scores = is_event_scores.transpose(1, 0).squeeze()

        labels_foreground = labels[:, :, :-1]
        labels_BCE, labels_evn = labels_foreground.max(-1)
        labels_event, _ = labels_evn.max(-1)

        loss_is_event = criterion(is_event_scores, labels_BCE.cuda())
        loss_event_class = criterion_event(event_scores, labels_event.cuda())
        loss = loss_is_event + loss_event_class

        acc = compute_accuracy_supervised(is_event_scores, event_scores, labels)
        accuracy.update(acc.item(), bs * 10)

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        losses.update(loss.item(), bs * 10)


    logger.info(f"\tOver Evaluation results (acc): {accuracy.avg:.4f}%.\t")

    '''Add loss in an epoch to Tensorboard'''
    if not eval_only:
        writer.add_scalar('Val_epoch_data/epoch_loss', losses.avg, epoch)
        writer.add_scalar('Val_epoch/Accuracy', accuracy.avg, epoch)

    logger.info(
        f"\tEvaluation results (acc): {accuracy.avg:.4f}%."
    )

    return accuracy.avg


def compute_accuracy_supervised(is_event_scores, event_scores, labels):

    _, targets = labels.max(-1)

    is_event_scores = is_event_scores.sigmoid()
    scores_pos_ind = is_event_scores > 0.5
    scores_mask = scores_pos_ind == 0
    _, event_class = event_scores.max(-1)
    pred = scores_pos_ind.long()
    pred *= event_class[:, None]

    pred[scores_mask] = 28

    correct = pred.eq(targets)
    correct_num = correct.sum().double()
    acc = correct_num * (100. / correct.numel())

    return acc

def save_checkpoint(state_dict, top1, task, epoch):
    model_name = f'{args.snapshot_pref}/model_epoch_{epoch}_top1_{top1:.3f}_task_{task}_best_model.pth.tar'
    torch.save(state_dict, model_name)

if __name__ == '__main__':
    main()
