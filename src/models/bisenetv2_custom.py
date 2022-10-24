import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------- Encoder ---------------------------
class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
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
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),  # not shown in paper
        )

        # 3. 1x1 Proj Conv (proj the dwconv output to low channel capacity)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
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
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
        )

        # 3. DWConv2 - 2 DWConvs for enlarging receptive field
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),  # not shown in paper
        )

        # 4. Proj
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )

        # Seperable 3x3 Conv for skip-conn
        self.conv2[1].last_bn = True
        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_chan, in_chan, kernel_size=3, stride=2,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.Conv2d(
                in_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
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

        self.S1S2 = StemBlock()  # 3 layer

        self.S3 = nn.Sequential(
            GELayerDown(16, 64),  # 4 layer
            GELayer(64, 64),  # 3 layer
        )

        self.S4 = nn.Sequential(
            GELayerDown(64, 128),
            GELayer(128, 128),
            GELayer(128, 128),
            GELayer(128, 128),
        )

        self.S5 = CEBlock()  # 2 layer

    def forward(self, x):  # (N, 3, 512, 512)
        feat2 = self.S1S2(x)  # (N, 16, 256, 256)
        feat3 = self.S3(feat2)  # (N, 64, 128, 128)
        feat4 = self.S4(feat3)  # (N, 128, 64, 64)
        feat5 = self.S5(feat4)  # (N, 128, 64, 64)

        return feat5


# -------------------------- Detail Branch ---------------------------
class DetailBranch(nn.Module):

    def __init__(self):
        super(DetailBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(3, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
        )

        self.S2 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2),
            ConvBNReLU(128, 128, 3, stride=1),
            ConvBNReLU(128, 128, 3, stride=1),
        )

    def forward(self, x):  # (N, 3, 512, 512)
        feat = self.S1(x)  # (N, 64, 256, 256)
        feat = self.S2(feat)  # (N, 128, 128, 128)

        return feat


# ---------------------- BGA ---------------------------
class BGALayer(nn.Module):
    '''fuse the complementary information from detail & encoder branches
       detail-branch-(N, 128, H, W) / encoder-branch-(N, 128, H/2, W/2)'''

    def __init__(self):
        super(BGALayer, self).__init__()
        # ---- Process Detail Branch ----
        # 1. 3x3 depth-wise conv & 1x1 conv
        self.left1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1, padding=0, bias=False),
        )  # (N, H, W, 128)

        # 2. 3x3 conv with stride=1 & 3x3 APooling
        self.left2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1, padding=1, bias=False),  # (N, H, W, 128)
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)  # (N, H/2, W/2, 128)
        )

        # ---- Process Encoder ----
        # 1. 3x3 conv & 2x2 up-sample
        self.right1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )  # (N, H/2, W/2, 128)
        self.up1 = nn.Upsample(scale_factor=2)  # (N, H, W, 128)

        # 2. 3x3 depth-wise conv & 1x1 conv
        self.right2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),  # (N, H/2, W/2, 128)
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1, padding=0, bias=False),  # (N, H/2, W/2, 128)
        )

        self.up2 = nn.Upsample(scale_factor=2)  # use before summation

        ##TODO: does this really has no relu?
        self.conv = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # not shown in paper
        )

    def forward(self, x_d, x_s):
        '''x_d: output of detail-branch (N, 128, h/4, w/4) - (N, 128, 128, 128)
           x_s: output of encoder-branch (N, 128, h/8, w/8) - (N, 128, 64, 64)'''
        dsize = x_d.size()[2:]

        # Detail Branch
        left1 = self.left1(x_d)  # (N, 128, h/4, w/4)
        left2 = self.left2(x_d)  # (N, 128, h/8, w/8)

        # Semantic Branch
        right1 = self.right1(x_s)  # (N, 128, h/8, w/8)
        right1 = self.up1(right1)  # (N, 128, h/4, w/4)
        right2 = self.right2(x_s)  # (N, 128, h/8, w/8)

        # Fuse
        left = left1 * torch.sigmoid(right1)  # (N, 128, h/4, w/4)

        right = left2 * torch.sigmoid(right2)  # (N, 128, h/8, w/8)
        right = self.up2(right)  # (N, 128, h/4, w/4)

        out = self.conv(left + right)  # (N, 128, h/4, w/4)

        return out


# ------------------------------- Decoder -------------------------------
class ConvBNReLU_Up(nn.Module):
    '''Input: (N, Input_ch, H, W) -> Output: (N, output_ch, 2H, 2W)'''

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(
            in_channel, out_channel, 3, stride=2, padding=1, output_padding=1, bias=False)
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

        self.layers.append(ConvBNReLU(64, 16))
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