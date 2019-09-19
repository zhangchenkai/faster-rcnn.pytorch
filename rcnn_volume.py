# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pdb
import pprint
import random
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler

from lib.model.faster_rcnn.vgg16 import vgg16
# from model.nms.nms_wrapper import nms
from lib.model.roi_layers import nms
from lib.model.rpn.bbox_transform import bbox_transform_inv
from lib.model.rpn.bbox_transform import clip_boxes
from lib.model.utils.config import cfg, cfg_from_file
from lib.model.utils.net_utils import save_checkpoint, clip_gradient
from lib.roi_data_layer.roibatchLoader import roibatchLoader
from lib.roi_data_layer.roidb import combined_roidb


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101',
                        default='vgg16', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs',
                        help='number of epochs to train',
                        default=40, type=int)
    # parser.add_argument('--disp_interval', dest='disp_interval',
    #                     help='number of iterations to display',
    #                     default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models',
                        default="saved_models", type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=4, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='cuda device id',
                        default='0', type=str)
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=8, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="adam", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--weight_decay', dest='weight_decay',
                        help='l2 weight decay',
                        default=0.0005, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

    # resume trained model
    parser.add_argument('--resume', help='resume checkpoint or not',
                        action='store_true')
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)

    # log and diaplay
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        help='whether use tensorboard',
                        action='store_true')

    parser.add_argument('--overlap_threshs', nargs='+', default=[0.5, 0.75], type=float)

    args = parser.parse_args()
    return args


class SubsetSampler(Sampler):
    def __init__(self, dataset, subset_ratio=1.0):

        assert 0 < subset_ratio <= 1
        num_sample = len(dataset)
        random.seed(321)
        if subset_ratio != 1:
            self.indices = random.sample(range(num_sample), int(num_sample * subset_ratio))
        else:
            self.indices = np.arange(num_sample)

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


if __name__ == '__main__':
    import os
    import pandas as pd

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    DATASET = 'fabric_binary'
    P_TYPE = 'P-%d'
    P_NUM = 15

    print('Called with args:')
    print(args)

    test_metrics_list = []
    for ratio in np.linspace(0, 1, 11)[1:]:
        # for ratio in [1.0, ]:
        # test_metrics_list.append({'model_name': 'subset ration = %g' % ratio})

        for p_id in range(1, P_NUM + 1):
            # for p_id in [5, ]:
            p_str = P_TYPE % p_id
            print('{0:#^64}'.format('Ratio-%g,' % ratio + p_str))

            # ===========================TRAIN============================
            # args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)
            args.cfg_file = "cfgs/fabric.yml"
            if args.cfg_file is not None:
                cfg_from_file(args.cfg_file)

            # args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5, 1, 2]']
            # if args.set_cfgs is not None:
            #     cfg_from_list(args.set_cfgs)

            # -- Note: Use validation set and disable the flipped to enable faster loading.
            if args.cuda:
                cfg.CUDA = True
            cfg.TRAIN.USE_FLIPPED = True
            cfg.USE_GPU_NMS = True if args.cuda else False
            print('Using config:')
            pprint.pprint(cfg)
            # np.random.seed(cfg.RNG_SEED)

            # train set
            imdb_name = DATASET + "_p%d_train_supervised" % p_id
            imdb, roidb, ratio_list, ratio_index = combined_roidb(imdb_name)
            train_size = len(roidb)
            print('{:d} roidb entries'.format(train_size))

            output_dir = os.path.join(args.save_dir, args.net, DATASET, 'ratio-%.2f' % ratio, 'pattern-%d' % p_id)
            os.makedirs(output_dir, exist_ok=True)

            dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size,
                                     imdb.num_classes, training=True)
            subset_sampler = SubsetSampler(dataset=dataset, subset_ratio=ratio)
            train_size = len(subset_sampler)
            print('subset {:d} roidb entries'.format(train_size))
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size,
                sampler=subset_sampler, num_workers=args.num_workers, pin_memory=False
            )

            # initilize the tensor holder here.
            im_data = torch.FloatTensor(1)
            im_info = torch.FloatTensor(1)
            num_boxes = torch.LongTensor(1)
            gt_boxes = torch.FloatTensor(1)

            # ship to cuda
            if args.cuda:
                im_data = im_data.cuda()
                im_info = im_info.cuda()
                num_boxes = num_boxes.cuda()
                gt_boxes = gt_boxes.cuda()

            # make variable
            im_data = Variable(im_data)
            im_info = Variable(im_info)
            num_boxes = Variable(num_boxes)
            gt_boxes = Variable(gt_boxes)

            # initilize the network here.
            if args.net == 'vgg16':
                fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
            else:
                fasterRCNN = None
                print("network is not defined")
                pdb.set_trace()

            fasterRCNN.create_architecture()

            lr = cfg.TRAIN.LEARNING_RATE = args.lr
            weight_decay = cfg.TRAIN.WEIGHT_DECAY = args.weight_decay
            # tr_momentum = cfg.TRAIN.MOMENTUM
            # tr_momentum = args.momentum

            if args.cuda:
                fasterRCNN.cuda()

            if args.optimizer == "adam":
                # optimizer = torch.optim.Adam(fasterRCNN.parameters(), lr=lr * 0.1)
                optimizer = torch.optim.Adam(fasterRCNN.parameters(), lr=lr, weight_decay=weight_decay)
            elif args.optimizer == "sgd":
                optimizer = torch.optim.SGD(fasterRCNN.parameters(), lr=lr, momentum=cfg.TRAIN.MOMENTUM, weight_decay=weight_decay)
            else:
                optimizer = None

            # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25, ], gamma=0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

            if args.resume:
                load_name = os.path.join(output_dir, 'pattern{}_s{}.pth'.format(p_id, args.session))
                print("loading checkpoint %s" % (load_name))
                checkpoint = torch.load(load_name)
                args.session = checkpoint['session']
                args.start_epoch = checkpoint['epoch']
                fasterRCNN.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr = optimizer.param_groups[0]['lr']
                if 'pooling_mode' in checkpoint.keys():
                    cfg.POOLING_MODE = checkpoint['pooling_mode']
                print("loaded checkpoint %s" % (load_name))

            if args.mGPUs:
                fasterRCNN = nn.DataParallel(fasterRCNN)

            iters_per_epoch = int(train_size / args.batch_size)

            if args.use_tfboard:
                from tensorboardX import SummaryWriter

                logger = SummaryWriter("logs")

            # setting to train mode
            fasterRCNN.train()
            for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
                scheduler.step()
                lr = optimizer.param_groups[0]['lr']

                start = time.time()

                steps = 0
                loss_temp = 0
                loss_rpn_cls = 0
                loss_rpn_box = 0
                loss_rcnn_cls = 0
                loss_rcnn_box = 0
                fg_cnt = 0
                bg_cnt = 0

                for data in dataloader:
                    with torch.no_grad():
                        im_data.resize_(data[0].size()).copy_(data[0])
                        im_info.resize_(data[1].size()).copy_(data[1])
                        gt_boxes.resize_(data[2].size()).copy_(data[2])
                        num_boxes.resize_(data[3].size()).copy_(data[3])

                    fasterRCNN.zero_grad()
                    rois, cls_prob, bbox_pred, \
                    rpn_loss_cls, rpn_loss_box, \
                    RCNN_loss_cls, RCNN_loss_bbox, \
                    rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

                    if args.mGPUs:
                        loss_rpn_cls += rpn_loss_cls.mean().item()
                        loss_rpn_box += rpn_loss_box.mean().item()
                        loss_rcnn_cls += RCNN_loss_cls.mean().item()
                        loss_rcnn_box += RCNN_loss_bbox.mean().item()
                    else:
                        loss_rpn_cls += rpn_loss_cls.item()
                        loss_rpn_box += rpn_loss_box.item()
                        loss_rcnn_cls += RCNN_loss_cls.item()
                        loss_rcnn_box += RCNN_loss_bbox.item()
                    fg_cnt += torch.sum(rois_label.data != 0)
                    bg_cnt += torch.sum(rois_label.data == 0)

                    loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
                    loss_temp += loss.item()

                    # backward
                    optimizer.zero_grad()
                    loss.backward()
                    if args.net == "vgg16":
                        clip_gradient(fasterRCNN, 10.)
                    optimizer.step()

                    steps += 1
                    end = time.time()

                loss_rpn_cls /= steps
                loss_rpn_box /= steps
                loss_rcnn_cls /= steps
                loss_rcnn_box /= steps
                loss_temp /= steps

                print("[session %d][epoch %2d] loss: %.4f, lr: %.2e" \
                      % (args.session, epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box
                    }
                    logger.add_scalars("logs_epoch_{}/losses".format(args.session), info, epoch)

            if args.epochs != 0:
                save_name = os.path.join(output_dir, 'pattern{}_s{}.pth'.format(p_id, args.session))
                save_checkpoint({
                    'session': args.session,
                    'epoch': epoch,
                    'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'pooling_mode': cfg.POOLING_MODE,
                    'class_agnostic': args.class_agnostic,
                }, save_name)
                print('save model: {}'.format(save_name))

            if args.use_tfboard:
                logger.close()

            # ===========================TEST============================
            cfg.TRAIN.USE_FLIPPED = False
            imdb_name = DATASET + "_p%d_test" % p_id
            imdb, roidb, ratio_list, ratio_index = combined_roidb(imdb_name, training=False)
            imdb.competition_mode(on=True)
            print('{:d} roidb entries'.format(len(roidb)))

            # initilize the tensor holder here.
            im_data = torch.FloatTensor(1)
            im_info = torch.FloatTensor(1)
            num_boxes = torch.LongTensor(1)
            gt_boxes = torch.FloatTensor(1)

            # ship to cuda
            if args.cuda:
                im_data = im_data.cuda()
                im_info = im_info.cuda()
                num_boxes = num_boxes.cuda()
                gt_boxes = gt_boxes.cuda()

            # make variable
            im_data = Variable(im_data)
            im_info = Variable(im_info)
            num_boxes = Variable(num_boxes)
            gt_boxes = Variable(gt_boxes)

            start = time.time()
            max_per_image = 100
            thresh = 0.0

            num_images = len(imdb.image_index)
            all_boxes = [[[] for _ in range(num_images)]
                         for _ in range(imdb.num_classes)]

            dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1,
                                     imdb.num_classes, training=False, normalize=False)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=False,
                num_workers=0, pin_memory=False
            )

            # _t = {'im_detect': time.time(), 'misc': time.time()}
            # det_file = os.path.join(output_dir, 'detections.pkl')

            fasterRCNN.eval()
            empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

            for i, data in enumerate(dataloader):
                with torch.no_grad():
                    im_data.resize_(data[0].size()).copy_(data[0])
                    im_info.resize_(data[1].size()).copy_(data[1])
                    gt_boxes.resize_(data[2].size()).copy_(data[2])
                    num_boxes.resize_(data[3].size()).copy_(data[3])

                det_tic = time.time()
                rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

                scores = cls_prob.data
                boxes = rois.data[:, :, 1:5]

                if cfg.TEST.BBOX_REG:
                    # Apply bounding-box regression deltas
                    box_deltas = bbox_pred.data
                    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                        # Optionally normalize targets by a precomputed mean and stdev
                        if args.class_agnostic:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                            box_deltas = box_deltas.view(1, -1, 4)
                        else:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                            box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

                    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                    pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
                else:
                    # Simply repeat the boxes, once for each class
                    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

                pred_boxes /= data[1][0][2].item()

                scores = scores.squeeze()
                pred_boxes = pred_boxes.squeeze()
                det_toc = time.time()
                detect_time = det_toc - det_tic
                misc_tic = time.time()

                for j in range(1, imdb.num_classes):
                    inds = torch.nonzero(scores[:, j] > thresh).view(-1)
                    # if there is det
                    if inds.numel() > 0:
                        cls_scores = scores[:, j][inds]
                        _, order = torch.sort(cls_scores, 0, True)
                        if args.class_agnostic:
                            cls_boxes = pred_boxes[inds, :]
                        else:
                            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                        # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                        cls_dets = cls_dets[order]
                        keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                        cls_dets = cls_dets[keep.view(-1).long()]
                        all_boxes[j][i] = cls_dets.cpu().numpy()
                    else:
                        all_boxes[j][i] = empty_array

                # Limit to max_per_image detections *over all classes*
                if max_per_image > 0:
                    image_scores = np.hstack([all_boxes[j][i][:, -1]
                                              for j in range(1, imdb.num_classes)])
                    if len(image_scores) > max_per_image:
                        image_thresh = np.sort(image_scores)[-max_per_image]
                        for j in range(1, imdb.num_classes):
                            keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                            all_boxes[j][i] = all_boxes[j][i][keep, :]

                misc_toc = time.time()
                nms_time = misc_toc - misc_tic
                # sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                #                  .format(i + 1, num_images, detect_time, nms_time))
                # sys.stdout.flush()

            # with open(det_file, 'wb') as f:
            #     pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

            ap_thresh = OrderedDict({'pattern': p_id, 'volume_ratio': ratio})
            aps = imdb.evaluate_detections(all_boxes, output_dir, overlap_threshs=args.overlap_threshs)
            ap_thresh.update(dict(zip(['AP@0.5', 'AP@0.75'], aps)))

            end = time.time()
            print("test time: %0.4fs" % (end - start))

            test_metrics_list.append(ap_thresh)

            # incremental result saving
            df = pd.DataFrame(test_metrics_list)
            df.to_csv('~/Desktop/FabricDataset/rcnn_volume.csv', index=False)
