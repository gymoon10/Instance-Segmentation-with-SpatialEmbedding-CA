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


# -------------------------- Detail Branch ---------------------------
class DetailBranch(nn.Module):
    '''shallow network compared to the encoder
         but can richly contain detailed information for each pixel by having more channel dimensions'''
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


# -------------------------- Transformer for Cross-Attention ------------------------
# References
# 1. Multi-Modality Cross Attention Network for Image and Sentence Matching. CVPR 2020
# 2. Restormer: Efficient Transformer for High-Resolution Image Restoration. arXiv:2111.09881, 2021.11

class GDFN_1(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN_1, self).__init__()

        hidden_channels = int(channels * expansion_factor)  # channel expansion
        # 1x1 conv to extend feature channel
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)

        # 3x3 DW Conv (groups=input_channels) -> each input channel is convolved with its own set of filters
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)

        # 1x1 conv to reduce channels back to original input dimension
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        '''HxWxC -> HxWxC'''
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        # Gating: the element-wise product of 2 parallel paths of linear transformation layers
        x = self.project_out(F.gelu(x1) * x2)

        return x

class MDTA_1(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA_1, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3,
                                  bias=False)  # DConv

        self.project_out_1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.project_out_2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.project_out_3 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.project_out_4 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)


    def forward(self, e, d):
        '''e, d: (N, C, H, W) - encoder/detail-branch features to fuse'''
        b, c, h, w = e.shape

        q_e, k_e, v_e = self.qkv_conv(self.qkv(e)).chunk(3, dim=1)  # (N, C, H, W)
        q_d, k_d, v_d = self.qkv_conv(self.qkv(d)).chunk(3, dim=1)  # (N, C, H, W)

        # divide the # of channels into heads & learn separate attention map
        q_e = q_e.reshape(b, self.num_heads, -1, h * w)  # (N, num_heads, C/num_heads, HW)
        k_e = k_e.reshape(b, self.num_heads, -1, h * w)
        v_e = v_e.reshape(b, self.num_heads, -1, h * w)

        q_d = q_d.reshape(b, self.num_heads, -1, h * w)  # (N, num_heads, C/num_heads, HW)
        k_d = k_d.reshape(b, self.num_heads, -1, h * w)
        v_d = v_d.reshape(b, self.num_heads, -1, h * w)

        q_e, k_e = F.normalize(q_e, dim=-1), F.normalize(k_e, dim=-1)
        q_d, k_d = F.normalize(q_d, dim=-1), F.normalize(k_d, dim=-1)

        # SA(Intra) - CxC Self Attention map instead of HWxHW (when num_heads=1)
        self_attn_e = torch.softmax(torch.matmul(q_e, k_e.transpose(-2, -1).contiguous()) * self.temperature,
                                    dim=-1)  # (N, num_heads, C/num_heads, C_num_heads)
        self_attn_d = torch.softmax(torch.matmul(q_d, k_d.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)

        intra_e = self.project_out_1(torch.matmul(self_attn_e, v_e).reshape(b, -1, h, w))  # (N, C, H, W)
        intra_d = self.project_out_2(torch.matmul(self_attn_d, v_d).reshape(b, -1, h, w))

        # CA(Inter) - CxC Cross Attention map instead of HWxHW (when num_heads=1)
        cross_attn_ed = torch.softmax(torch.matmul(q_e, k_d.transpose(-2, -1).contiguous()) * self.temperature,
                                      dim=-1)  # (N, num_heads, C/num_heads, C_num_heads)
        cross_attn_de = torch.softmax(torch.matmul(q_d, k_e.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        cross_attn_de = cross_attn_de.squeeze(0)

        inter_ed = self.project_out_3(torch.matmul(cross_attn_ed, v_d).reshape(b, -1, h, w))  # (N, C, H, W)
        inter_de = self.project_out_4(torch.matmul(cross_attn_de, v_e).reshape(b, -1, h, w))
        # out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))

        return intra_e, intra_d, inter_ed, inter_de


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1_1 = nn.LayerNorm(channels)
        self.norm1_2 = nn.LayerNorm(channels)

        self.attn = MDTA_1(channels, num_heads)

        self.norm2_1 = nn.LayerNorm(channels)
        self.norm2_2 = nn.LayerNorm(channels)

        # parallel GDFNs
        self.ffn_1 = GDFN_1(channels, expansion_factor)
        self.ffn_2 = GDFN_1(channels, expansion_factor)


    def forward(self, e, d):
        '''e: upsampled encoder feature (N, 128, 128, 128)
           d: Detail feature (N, 128, 128, 128)'''
        b, c, h, w = e.shape

        # SA feature-e, SA feature-d, CA feature-(query=e, key=d), CA featue(query=d, key=e)
        _, _, cross_ed, cross_de = self.attn(
            self.norm1_1(e.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
            .contiguous().reshape(b, c, h, w),
            self.norm1_2(d.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
            .contiguous().reshape(b, c, h, w))

        e = e + cross_ed
        d = d + cross_de

        # GDFNs
        ca_ed = e + self.ffn_1(self.norm2_1(e.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                               .contiguous().reshape(b, c, h, w))

        ca_de = d + self.ffn_2(self.norm2_2(d.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                               .contiguous().reshape(b, c, h, w))

        return ca_ed, ca_de


# ------------------------------ Decoder -------------------------------
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
            16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=False)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


# --------------------- Network ---------------------------
class Net(nn.Module):

    def __init__(self, num_classes):  # use encoder to pass pretrained encoder
        super().__init__()

        self.encoder = Encoder()
        self.detail_branch = DetailBranch()
        self.up = ConvBNReLU_Up(128, 128)
        self.fuse_transform = TransformerBlock(channels=128, num_heads=4, expansion_factor=2.66)  # channels % num_heads must be 0
        self.decoder = Decoder(num_classes)

    def forward(self, input): # input : (N, 3, 512, 512)
        encoded_feature = self.encoder(input)  # (N, 128, 64, 64)
        encoded_feature = self.up(encoded_feature)  # (N, 128, 128, 128)
        detail_feature = self.detail_branch(input)  # (N, 128, 128, 128)

        # CA feature (query=encoder feature, key=detail feature)
        ca_feature_ed, _ = self.fuse_transform(encoded_feature, detail_feature)  # (N, 128. 128, 128)

        return self.decoder.forward(ca_feature_ed)  # (N, num_classes, 512, 512)