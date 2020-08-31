import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F

class VGG16Extractor(nn.Module):
    def __init__(self):
        super(VGG16Extractor, self).__init__()
        self.select = {
            '1': 'conv1_1',  # [batch_size, 64, 224, 224]
            '3': 'conv1_2',  # [batch_size, 64, 224, 224]
            '4': 'pooled_1',  # [batch_size, 64, 112, 112]
            '6': 'conv2_1',  # [batch_size, 128, 112, 112]
            '8': 'conv2_2',  # [batch_size, 128, 112, 112]
            '9': 'pooled_2',  # [batch_size, 128, 56, 56]
            '11': 'conv3_1',  # [batch_size, 256, 56, 56]
            '13': 'conv3_2',  # [batch_size, 256, 56, 56]
            '15': 'conv3_3',  # [batch_size, 256, 56, 56]
            '16': 'pooled_3',  # [batch_size, 256, 28, 28]
            '18': 'conv4_1',  # [batch_size, 512, 28, 28]
            '20': 'conv4_2',  # [batch_size, 512, 28, 28]
            '22': 'conv4_3',  # [batch_size, 512, 28, 28]
            '23': 'pooled_4',  # [batch_size, 512, 14, 14]
            '25': 'conv5_1',  # [batch_size, 512, 14, 14]
            '27': 'conv5_2',  # [batch_size, 512, 14, 14]
            '29': 'conv5_3',  # [batch_size, 512, 14, 14]
            '30': 'pooled_5',  # [batch_size , 512, 7, 7]
        }
        self.vgg = torchvision.models.vgg16(pretrained=True).features

    def forward(self, x):
        output = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                output[self.select[name]] = x
        return output

class LandmarkUpsample(nn.Module):
    def __init__(self, in_channel=256):
        super(LandmarkUpsample, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, 1, 1, 0)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 1)
        self.upconv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.conv7 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv8 = nn.Conv2d(32, 32, 3, 1, 1)
        self.upconv3 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        self.conv9 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv10 = nn.Conv2d(16, 8, 1, 1, 0)

        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.upconv1(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.upconv2(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.relu(self.upconv3(x))
        x = self.relu(self.conv9(x))
        x = self.conv10(x)
        return x

class MessagePass(nn.Module):
    def __init__(self, in_channel=128):
        super(MessagePass, self).__init__()
        self.inplanes = 128
        self.ca = ChannelAttention(self.inplanes)
        self.sa = SpatialAttention()
        self.res_layer = self._make_layer(Bottleneck, 128, 4)
        self.conv = nn.Conv2d(512, 128, 1, 1)
        self.bn = nn.BatchNorm2d(128)
        # self.ca1 = ChannelAttention(512)
        # self.sa1 = SpatialAttention()
        self.relu = nn.ReLU(inplace=True)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        x = self.res_layer(x)
        # x = self.ca1(x) * x
        # x = self.sa1(x) * x
        x = self.relu(self.bn(self.conv(x)))
        # x = self.relu(self.conv1(x))
        # x = self.relu(self.conv2(x))
        # x = self.conv3(x)
        # x = self.relu(self.bn1(self.conv1(x)))
        # x = self.relu(self.bn2(self.conv2(x)))
        return x

class TreeNet(nn.Module):
    def __init__(self):
        super(TreeNet, self).__init__()
        self.message_pass = MessagePass()
        self.conv = transition_block(256, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = x
        conv1_a = conv
        conv2_a = conv
        conv3_a = conv
        conv4_a = conv
        conv5_a = conv
        conv6_a = conv
        conv7_a = conv
        conv8_a = conv
        # root --> leaf
        conv1to3 = self.message_pass(conv1_a)
        conv1to5 = self.message_pass(conv1_a)
        conv1to2 = self.message_pass(conv1_a)

        conv2to4 = self.message_pass(conv2_a)
        conv2to6 = self.message_pass(conv2_a)

        conv5to6 = self.message_pass(conv5_a)
        conv5to7 = self.message_pass(conv5_a)

        conv6to8 = self.message_pass(conv6_a)

        conv7to8 = self.message_pass(conv7_a)

        conv1_a_plus = conv1_a
        conv2_a_plus = self.relu(conv2_a + conv1to2)
        conv3_a_plus = self.relu(conv3_a + conv1to3)
        conv4_a_plus = self.relu(conv4_a + conv2to4)
        conv5_a_plus = self.relu(conv5_a + conv1to5)
        conv6_a_plus = self.relu(conv6_a + conv2to6 + conv5to6)
        conv7_a_plus = self.relu(conv7_a + conv5to7)
        conv8_a_plus = self.relu(conv8_a + conv6to8 + conv7to8)

        conv1_b = conv
        conv2_b = conv
        conv3_b = conv
        conv4_b = conv
        conv5_b = conv
        conv6_b = conv
        conv7_b = conv
        conv8_b = conv
        # leaf --> root
        conv8to6 = self.message_pass(conv8_b)
        conv8to7 = self.message_pass(conv8_b)

        conv7to5 = self.message_pass(conv7_b)

        conv6to2 = self.message_pass(conv6_b)
        conv6to5 = self.message_pass(conv6_b)

        conv5to1 = self.message_pass(conv5_b)

        conv4to2 = self.message_pass(conv4_b)

        conv3to1 = self.message_pass(conv3_b)

        conv2to1 = self.message_pass(conv2_b)

        conv8_b_plus = conv8_b
        conv7_b_plus = self.relu(conv7_b + conv8to7)
        conv6_b_plus = self.relu(conv6_b + conv8to6)
        conv5_b_plus = self.relu(conv5_b + conv6to5 + conv7to5)
        conv4_b_plus = conv4_b
        conv3_b_plus = conv3_b
        conv2_b_plus = self.relu(conv2_b + conv4to2 + conv6to2)
        conv1_b_plus = self.relu(conv1_b + conv2to1 + conv3to1 + conv5to1)

        conv1cat = torch.cat((conv1_a_plus, conv1_b_plus), dim=1)
        conv2cat = torch.cat((conv2_a_plus, conv2_b_plus), dim=1)
        conv3cat = torch.cat((conv3_a_plus, conv3_b_plus), dim=1)
        conv4cat = torch.cat((conv4_a_plus, conv4_b_plus), dim=1)
        conv5cat = torch.cat((conv5_a_plus, conv5_b_plus), dim=1)
        conv6cat = torch.cat((conv6_a_plus, conv6_b_plus), dim=1)
        conv7cat = torch.cat((conv7_a_plus, conv7_b_plus), dim=1)
        conv8cat = torch.cat((conv8_a_plus, conv8_b_plus), dim=1)

        conv9_1 = self.conv(conv1cat)
        conv9_2 = self.conv(conv2cat)
        conv9_3 = self.conv(conv3cat)
        conv9_4 = self.conv(conv4cat)
        conv9_5 = self.conv(conv5cat)
        conv9_6 = self.conv(conv6cat)
        conv9_7 = self.conv(conv7cat)
        conv9_8 = self.conv(conv8cat)

        conv9 = torch.cat((conv9_1, conv9_2, conv9_3, conv9_4, conv9_5, conv9_6, conv9_7, conv9_8), dim=1)
        allconv = torch.cat((conv1cat, conv2cat, conv3cat, conv4cat, conv5cat, conv6cat, conv7cat, conv8cat), dim=1)
        return allconv

def transition_block(in_channels, out_channels):
    blk = nn.Sequential(nn.BatchNorm2d(in_channels),
                        nn.ReLU(),
                        nn.Conv2d(in_channels, out_channels, kernel_size=1))
    return blk

class StructNet(nn.Module):
    def __init__(self):
        super(StructNet, self).__init__()
        self.feature_extractor = VGG16Extractor()
        self.conv = nn.Conv2d(512, 4096, 7, 1, 3)
        self.fcn6 = nn.Conv2d(4096, 128, 1, 1, 0)
        self.bn = nn.BatchNorm2d(4096)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)

        self.upsampling = LandmarkUpsample(512)

        self.transition_block = transition_block(2048, 512)
        self.treenet = TreeNet()

    def forward(self, sample):
        vgg16_output = self.feature_extractor(sample['image'])
        lm_feature = vgg16_output['conv4_3']
        fcn6_feature = self.dropout(self.relu(self.bn(self.conv(lm_feature))))
        fcn6_feature = self.relu(self.fcn6(fcn6_feature))
        lm_feature = self.treenet(fcn6_feature)
        tree_feature = lm_feature
        lm_feature = self.transition_block(lm_feature)
        lm_pos_map = self.upsampling(lm_feature)
        # lm_pos_map = lm_feature
        return {'lm_pos_map': lm_pos_map, 'tree_feature': tree_feature, 'fcn6_feature': fcn6_feature}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



