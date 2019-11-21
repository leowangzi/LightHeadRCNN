import torch as t
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn import _fasterRCNN


def channel_shuffle(x, groups=2):
    bat_size, channels, w, h = x.shape
    group_c = channels // groups
    x = x.view(bat_size, groups, group_c, w, h)
    x = t.transpose(x, 1, 2).contiguous()
    x = x.view(bat_size, -1, w, h)
    return x


# used in the block
def conv_1x1_bn(in_c, out_c, stride=1):
    return nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride, 0, bias=False),
                         nn.BatchNorm2d(out_c), nn.ReLU(True))


def conv_bn(in_c, out_c, stride=2):
    return nn.Sequential(nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(out_c), nn.ReLU(True))


class ShuffleBlock(nn.Module):
    def __init__(self, in_c, out_c, downsample=False):
        super(ShuffleBlock, self).__init__()
        self.downsample = downsample
        half_c = out_c // 2
        if downsample:
            self.branch1 = nn.Sequential(
                # 3*3 dw conv, stride = 2
                nn.Conv2d(in_c, in_c, 3, 2, 1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                # 1*1 pw conv
                nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True))

            self.branch2 = nn.Sequential(
                # 1*1 pw conv
                nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True),
                # 3*3 dw conv, stride = 2
                nn.Conv2d(half_c, half_c, 3, 2, 1, groups=half_c, bias=False),
                nn.BatchNorm2d(half_c),
                # 1*1 pw conv
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True))
        else:
            # in_c = out_c
            assert in_c == out_c

            self.branch2 = nn.Sequential(
                # 1*1 pw conv
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True),
                # 3*3 dw conv, stride = 1
                nn.Conv2d(half_c, half_c, 3, 1, 1, groups=half_c, bias=False),
                nn.BatchNorm2d(half_c),
                # 1*1 pw conv
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True))

    def forward(self, x):
        out = None
        if self.downsample:
            # if it is downsampling, we don't need to do channel split
            out = t.cat((self.branch1(x), self.branch2(x)), 1)
        else:
            # channel split
            channels = x.shape[1]
            c = channels // 2
            x1 = x[:, :c, :, :]
            x2 = x[:, c:, :, :]
            out = t.cat((x1, self.branch2(x2)), 1)
        return channel_shuffle(out, 2)


class ShuffleNet2(nn.Module):
    def __init__(self, num_classes=2, input_size=224, net_type=1):
        super(ShuffleNet2, self).__init__()
        assert input_size % 32 == 0  # 因为一共会下采样32倍

        self.stage_repeat_num = [4, 8, 4]
        if net_type == 0.5:
            self.out_channels = [3, 24, 48, 96, 192, 1024]
        elif net_type == 1:
            self.out_channels = [3, 24, 116, 232, 464, 1024]
        elif net_type == 1.5:
            self.out_channels = [3, 24, 176, 352, 704, 1024]
        elif net_type == 2:
            self.out_channels = [3, 24, 244, 488, 976, 2948]
        else:
            print("the type is error, you should choose 0.5, 1, 1.5 or 2")

        # let's start building layers
        self.conv1 = nn.Conv2d(3, self.out_channels[1], 3, 2, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        in_c = self.out_channels[1]

        self.features = []
        for stage_idx in range(len(self.stage_repeat_num)):
            out_c = self.out_channels[2 + stage_idx]
            repeat_num = self.stage_repeat_num[stage_idx]
            for i in range(repeat_num):
                if i == 0:
                    self.features.append(
                        ShuffleBlock(in_c, out_c, downsample=True))
                else:
                    self.features.append(
                        ShuffleBlock(in_c, in_c, downsample=False))
                in_c = out_c
        in_c = self.out_channels[-2]
        out_c = self.out_channels[-1]
        self.conv5 = conv_1x1_bn(in_c, out_c, 1)
        self.features.append(self.conv5)
        self.features = nn.Sequential(*self.features)

        self.g_avg_pool = nn.AvgPool2d(kernel_size=(int)(
            input_size / 32))  # 如果输入的是224，则此处为7

        # fc layer
        self.classifier = nn.Linear(out_c, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv5(x)
        x = self.g_avg_pool(x)
        x = x.view(-1, self.out_channels[-1])
        x = self.classifier(x)
        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv5(x)  # ?*?*1024
        # ---used for classification---
        x = self.g_avg_pool(x)
        x = x.view(-1, self.out_channels[-1])
        x = self.classifier(x)
        return x


class shufflenetv2(_fasterRCNN):
    def __init__(self,
                 classes,
                 pretrained=False,
                 class_agnostic=False,
                 lighthead=False):
        # self.model_path = 'data/pretrained_model/mobilenet_v2.pth.tar'
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        self.lighthead = lighthead
        self.dout_base_model = 464  # 改成464
        if self.lighthead:
            self.dout_lh_base_model = 1024  # 改成1024

        _fasterRCNN.__init__(self,
                             classes,
                             class_agnostic,
                             lighthead,
                             compact_mode=False)

    def _init_module(self):
        shufflenet = ShuffleNet2()
        if self.pretrained:
            print("not impelemented!")
            exit()

        # shufflenet.classifier = nn.Sequential(*list(shufflenet.features._modules.values())[-1:]) # ???

        # Build shufflenetv2
        self.RCNN_base = nn.Sequential(*(
            list(mobilenet.features._modules.values())))

        if self.lighthead:
            self.RCNN_top = nn.Sequential(nn.Linear(490 * 7 * 7, 2048),
                                          nn.ReLU(True))
        else:
            print("Not impelemented!")
            exit()

        c_in = 2048  # 如果不使用light-head，自行设计

        self.RCNN_cls_score = nn.Linear(c_in, self.num_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(c_in, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(c_in, 4 * self.num_classes)

        def _head_to_tail(self, pool5):
            if self.lighthead:
                pool5_flat = pool5.view(pool5.size(0), -1)
                fc7 = self.RCNN_top(
                    pool5_flat)  # or two large fully-connected layers
            else:
                print(pool5.shape)
                fc7 = self.RCNN_top(pool5)
                fc7 = fc7.view(fc7.size(0), -1)
            return fc7
