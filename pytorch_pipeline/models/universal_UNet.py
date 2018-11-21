import torch
from torch import nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.nn import Sequential
from collections import OrderedDict
import torchvision
from torch.nn import functional as F
from pretrainedmodels import se_resnext50_32x4d


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel//reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel//reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())


class DecoderBlockV(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)

            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class DecoderCenter(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderCenter, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
        nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels)

            )

    def forward(self, x):
        return self.block(x)


class GetEncoder:
    def __init__(self, encoder_name, encoder_depth, num_filters, pretrained):
        self.encoder_name = encoder_name
        self.num_filters = num_filters
        self.encoder_depth = encoder_depth
        self.pretrained = pretrained

    def get_encoder(self):
        if self.encoder_name == 'ResNet':
            return self._get_resnet()
        elif self.encoder_name == 'SeResNext':
            return self._get_serresnext()
        elif self.encoder_name == 'DenseNet':
            return self._get_densenet()
        else:
            raise NotImplementedError(f'Encoder {self.encoder_name} is not implemented')

    def _get_resnet(self):
        if self.encoder_depth == 34:
            encoder = torchvision.models.resnet34(pretrained=self.pretrained)
            bottom_channel_nr = 512
        elif self.encoder_depth == 101:
            encoder = torchvision.models.resnet101(pretrained=self.pretrained)
            bottom_channel_nr = 2048
        elif self.encoder_depth == 152:
            encoder = torchvision.models.resnet152(pretrained=self.pretrained)
            bottom_channel_nr = 2048

        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

        pool = nn.MaxPool2d(2, 2)

        conv1 = nn.Sequential(encoder.conv1,
                              encoder.bn1,
                              encoder.relu,
                              pool)
        conv2 = encoder.layer1
        conv3 = encoder.layer2
        conv4 = encoder.layer3
        conv5 = encoder.layer4
        return {'conv1': conv1, 'conv2': conv2, 'conv3': conv3, 'conv4': conv4, 'conv5': conv5,
                'bottom_channel_nr': bottom_channel_nr}

    def _get_serresnext(self):

        encoder = se_resnext50_32x4d(num_classes=1000, pretrained='imagenet')
        bottom_channel_nr = 2048

        conv1 = nn.Sequential(encoder.layer0, SCSEBlock(self.num_filters * 2))
        conv2 = nn.Sequential(encoder.layer1, SCSEBlock(self.num_filters * 8))
        conv3 = nn.Sequential(encoder.layer2, SCSEBlock(self.num_filters * 16))
        conv4 = nn.Sequential(encoder.layer3, SCSEBlock(self.num_filters * 32))
        conv5 = nn.Sequential(encoder.layer4, SCSEBlock(self.num_filters * 64))

        return {'conv1': conv1, 'conv2': conv2, 'conv3': conv3, 'conv4': conv4, 'conv5': conv5,
                'bottom_channel_nr': bottom_channel_nr}

    def _get_densenet(self):
        if self.encoder_depth == 121:
            encoder = torchvision.models.densenet121(pretrained=self.pretrained).features
            bottom_channel_nr = 1024
            w1 = 1088
            w2 = 576
            w3 = 320
        elif self.encoder_depth == 161:
            encoder = torchvision.models.densenet161(pretrained=self.pretrained).features
            bottom_channel_nr = 2208
        elif self.encoder_depth == 169:
            encoder = torchvision.models.densenet169(pretrained=self.pretrained).features
            bottom_channel_nr = 1664
        elif self.encoder_depth == 201:
            encoder = torchvision.models.densenet201(pretrained=self.pretrained).features
            bottom_channel_nr = 1920
            w1 = 1856
            w2 = 576
            w3 = 320
        else:
            raise NotImplementedError('only 121, 161, 169, 201 version of DenseNet are implemented')

        conv1 = nn.Sequential(encoder.conv0,
                                   encoder.norm0,
                                   encoder.relu0,
                                   encoder.pool0)

        conv2 = nn.Sequential(encoder.denseblock1, encoder.transition1)
        conv3 = nn.Sequential(encoder.denseblock2, encoder.transition2)
        conv4 = nn.Sequential(encoder.denseblock3,  encoder.transition3)
        conv5 = nn.Sequential(encoder.denseblock4,  encoder.norm5)
        return {'conv1': conv1, 'conv2': conv2, 'conv3': conv3, 'conv4': conv4, 'conv5': conv5,
                'bottom_channel_nr': bottom_channel_nr}


class UNet(nn.Module):
    def __init__(self, encoder_name, encoder_depth=50, num_classes=2, pretrained=True, num_filters=32, dropout_2d=0.2,
                 is_deconv=True, SCSE=False):
        super().__init__()
        self.dropout_2d = dropout_2d
        self.encoder = GetEncoder(encoder_name, encoder_depth=encoder_depth, pretrained=pretrained, num_filters=num_filters).get_encoder()
        self.decoder = GetDecoder(num_filters, self.encoder['bottom_channel_nr'], is_deconv, SCSE,
                                  is_densenet=(encoder_name == 'DenseNet')).get_decoder()
        self.final = nn.Conv2d(num_filters * 2, num_classes, kernel_size=1)
        self.dec0 = ConvRelu(num_filters * 10, num_filters * 2)

    def forward(self, x):
        conv1 = self.encoder['conv1'](x)
        conv2 = self.encoder['conv2'](conv1)
        conv3 = self.encoder['conv3'](conv2)
        conv4 = self.encoder['conv4'](conv3)
        conv5 = self.encoder['conv5'](conv4)
        center = self.decoder['center'](conv5)
        dec5 = self.decoder['dec5'](torch.cat([center, conv5], 1))
        dec4 = self.decoder['dec4'](torch.cat([dec5, conv4], 1))
        dec3 = self.decoder['dec3'](torch.cat([dec4, conv3], 1))
        dec2 = self.decoder['dec2'](torch.cat([dec3, conv2], 1))
        dec1 = self.decoder['dec1'](dec2)

        hypercolumn = torch.cat((
            dec1,
            F.upsample(dec2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(dec3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(dec4, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(dec5, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)

        dec0 = self.dec0(F.dropout2d(hypercolumn, p=self.dropout_2d))

        return self.final(dec0)


class GetDecoder:
    def __init__(self, num_filters, bottom_channel_nr, is_deconv, add_SCSE, is_densenet):
        self.num_filters = num_filters
        self.bottom_channel_nr = bottom_channel_nr
        self.is_deconv = is_deconv
        self.add_SCSE = add_SCSE
        self.is_densenet = is_densenet

    def get_decoder(self):
        center = DecoderCenter(self.bottom_channel_nr, self.num_filters * 8 * 2, self.num_filters * 8,
                               False)

        dec5 = DecoderBlockV(self.bottom_channel_nr + self.num_filters * 8, self.num_filters * 8 * 2,
                             self.num_filters * 2, self.is_deconv)
        if self.is_densenet:
            in_ch_4 = self.bottom_channel_nr + self.num_filters * 2
        else:
            in_ch_4 = self.bottom_channel_nr // 2 + self.num_filters * 2
        dec4 = DecoderBlockV(in_ch_4, self.num_filters * 8,
                             self.num_filters * 2, self.is_deconv)
        if self.is_densenet:
            in_ch_3 = self.bottom_channel_nr // 2 + self.num_filters * 2
        else:
            in_ch_3 = self.bottom_channel_nr // 4 + self.num_filters * 2

        dec3 = DecoderBlockV(in_ch_3, self.num_filters * 4,
                             self.num_filters * 2, self.is_deconv)
        if self.is_densenet:
            in_ch_2 = self.bottom_channel_nr // 4 + self.num_filters * 2
        else:
            in_ch_2 = self.bottom_channel_nr // 8 + self.num_filters * 2
        dec2 = DecoderBlockV(in_ch_2, self.num_filters * 2,
                             self.num_filters * 2, self.is_deconv)
        dec1 = DecoderBlockV(self.num_filters * 2, self.num_filters, self.num_filters * 2, self.is_deconv)
        if self.add_SCSE:
            dec5 = nn.Sequential(dec5, SCSEBlock(self.num_filters * 2))
            dec4 = nn.Sequential(dec4, SCSEBlock(self.num_filters * 2))
            dec3 = nn.Sequential(dec3, SCSEBlock(self.num_filters * 2))
            dec2 = nn.Sequential(dec2, SCSEBlock(self.num_filters * 2))
            dec1 = nn.Sequential(dec1, SCSEBlock(self.num_filters * 2))

        return {'dec5': dec5, 'dec4': dec4, 'dec3': dec3, 'dec2': dec2, 'dec1': dec1, 'center': center}





