# from torch import nn
# from .utils import load_state_dict_from_url

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn import _fasterRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

# __all__ = ['MobileNetV2', 'mobilenet_v2']
__all__ = ['mobilenetv2_o']

model_urls = {
    'mobilenet_v2':
    'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size,
                      stride,
                      padding,
                      groups=groups,
                      bias=False), nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim,
                       hidden_dim,
                       stride=stride,
                       groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(
                inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(
                                 inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult,
                                        round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel,
                          output_channel,
                          stride,
                          expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(
            ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        features.append(nn.AvgPool2d(224 / 32))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x = self.features(x)
        # x = x.mean([2, 3])
        # x = self.classifier(x)
        # return x
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x


class mobilenetv2_o(_fasterRCNN):
    def __init__(self,
                 classes,
                 pretrained=False,
                 class_agnostic=False,
                 lighthead=False):
        self.model_path = 'data/pretrained_model/mobilenet_v2-b0353104.pth'
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        self.lighthead = lighthead
        self.dout_base_model = 320
        if self.lighthead:
            self.dout_lh_base_model = 1280

        _fasterRCNN.__init__(self,
                             classes,
                             class_agnostic,
                             lighthead,
                             compact_mode=True)

    def _init_modules(self):
        mobilenet = MobileNetV2()

        if self.pretrained == True:
            print("Loading pretrained weights from %s" % (self.model_path))
            if torch.cuda.is_available():
                state_dict = torch.load(self.model_path)
            else:
                state_dict = torch.load(
                    self.model_path, map_location=lambda storage, loc: storage)

            mobilenet.load_state_dict({
                k: v
                for k, v in state_dict.items() if k in mobilenet.state_dict()
            })

        mobilenet.classifier = nn.Sequential(
            *list(mobilenet.features._modules.values())[-2:-1])

        # Build mobilenet.
        self.RCNN_base = nn.Sequential(
            *list(mobilenet.features._modules.values())[:-2])

        # Fix Layers
        if self.pretrained:
            for layer in range(len(self.RCNN_base)):
                for p in self.RCNN_base[layer].parameters():
                    p.requires_grad = False

        if self.lighthead:
            self.lighthead_base = mobilenet.classifier
            self.RCNN_top = nn.Sequential(nn.Linear(490 * 7 * 7, 2048),
                                          nn.ReLU(inplace=True))
        else:
            self.RCNN_top = mobilenet.classifier

        c_in = 2048 if self.lighthead else 1280 * 7 * 7

        self.RCNN_cls_score = nn.Linear(c_in, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(c_in, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(c_in, 4 * self.n_classes)

    def _head_to_tail(self, pool5):
        if self.lighthead:
            pool5_flat = pool5.view(pool5.size(0), -1)
            fc7 = self.RCNN_top(pool5_flat)  # or two large fully-connected layers
        else:
            fc7 = self.RCNN_top(pool5)
            fc7 = fc7.view(fc7.size(0), -1)
        return fc7