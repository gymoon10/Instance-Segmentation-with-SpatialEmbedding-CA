import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------- Encoder ---------------------------
class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_chan, out_chan, kernel_size=ks, stride=stride,
            padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)

        return feat


class StemBlock(nn.Module):
    '''2 different downsampling outputs are concatenated'''

    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(3, 16, 3, stride=1)
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            ConvBNReLU(8, 16, 3, stride=2),
        )
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(32, 16, 3, stride=1)

    def forward(self, x):  # (N, 3, 512, 512)
        feat = self.conv(x)  # (N, 16, 512, 512)

        feat_left = self.left(feat)  # (N, 16, 256, 256)
        feat_right = self.right(feat)  # (N, 16, 256, 256)

        feat = torch.cat([feat_left, feat_right], dim=1)  # (N, 32, 356, 256)
        feat = self.fuse(feat)  # (N, 16, 256, 256)

        return feat


class CEBlock(nn.Module):
    '''Context Embedding
       nput: (N, C, H, W) -> Output: (N, C, H, W)'''

    def __init__(self):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(128)
        self.conv_gap = ConvBNReLU(128, 128, 1, stride=1, padding=0)
        # TODO: in paper here is naive conv2d, no bn-relu
        self.conv_last = ConvBNReLU(128, 128, 3, stride=1)

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)

        return feat


class GELayer(nn.Module):
    '''Output_chanel must be same as input_channel
       Input: (N, C, H, W) -> Output: (N, C, H, W)'''

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayer, self).__init__()
        mid_chan = in_chan * exp_ratio  # expansion
        # 1. 3x3 Conv - aggregates feature responses
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)

        # 2. 3x3 DWConv - expand to higher dimension space
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=in_chan, bias=True),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),  # not shown in paper
        )

        # 3. 1x1 Proj Conv (proj the dwconv output to low channel capacity)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=True),
            nn.BatchNorm2d(out_chan),
        )

        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)

        feat = feat + x
        feat = self.relu(feat)

        return feat


class GELayerDown(nn.Module):
    '''Input: (N, C_in, H, W) -> Output: (N, C_out, H/2, W/2)
       Use stride=2 & adopt two 3x3 DWConv'''

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerDown, self).__init__()
        mid_chan = in_chan * exp_ratio
        # 1. 3x3 Conv
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)

        # 2. DWConv1
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=2,  # stride=2
                padding=1, groups=in_chan, bias=True),
            nn.BatchNorm2d(mid_chan),
        )

        # 3. DWConv2 - 2 DWConvs for enlarging receptive field
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=mid_chan, bias=True),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),  # not shown in paper
        )

        # 4. Proj
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=True),
            nn.BatchNorm2d(out_chan),
        )

        # Seperable 3x3 Conv for skip-conn
        self.conv2[1].last_bn = True
        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_chan, in_chan, kernel_size=3, stride=2,
                padding=1, groups=in_chan, bias=True),
            nn.BatchNorm2d(in_chan),
            nn.Conv2d(
                in_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=True),
            nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)

        shortcut = self.shortcut(x)

        feat = feat + shortcut
        feat = self.relu(feat)

        return feat


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.S1S2 = StemBlock()

        self.S3 = nn.Sequential(
            GELayerDown(16, 64),
            GELayer(64, 64),
        )

        self.S4 = nn.Sequential(
            GELayerDown(64, 128),
            GELayer(128, 128),
            GELayer(128, 128),
            GELayer(128, 128),
        )

        self.S5 = CEBlock()

    def forward(self, x):  # (N, 3, 512, 512)
        feat2 = self.S1S2(x)  # (N, 16, 256, 256)
        feat3 = self.S3(feat2)  # (N, 64, 128, 128)
        feat4 = self.S4(feat3)  # (N, 128, 64, 64)
        feat5 = self.S5(feat4)  # (N, 128, 64, 64)

        return feat5


# ------------------------------- Decoder -------------------------------
class ConvBNReLU_Up(nn.Module):
    '''Input: (N, Input_ch, H, W) -> Output: (N, output_ch, 2H, 2W)'''

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(
            in_channel, out_channel, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        up_feat = self.up_conv(x)
        up_feat = self.bn(up_feat)
        up_feat = self.relu(up_feat)
        return up_feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)


class Decoder(nn.Module):
    '''small (not symmetric) decoder that upsamples encoder's output
    by fine-tuning the details'''

    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(ConvBNReLU_Up(128, 64))
        self.layers.append(GELayer(64, 64))
        self.layers.append(GELayer(64, 64))

        self.layers.append(ConvBNReLU_Up(64, 16))
        self.layers.append(GELayer(16, 16))
        self.layers.append(GELayer(16, 16))

        self.output_conv = nn.ConvTranspose2d(
            16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output