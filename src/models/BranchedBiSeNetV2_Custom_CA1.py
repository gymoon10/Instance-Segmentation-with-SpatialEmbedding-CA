import torch
import torch.nn as nn
import models.bisenetv2_custom_ca1 as network


class BranchedBiSeNetV2_Custom_CA1(nn.Module):
    '''shared encoder + 2 branched decoders'''

    def __init__(self, num_classes, encoder=None):
        super().__init__()

        self.num_classes = num_classes
        print('Creating Branched BiSeNetV2_Custom_CA1 with {} classes'.format(num_classes))

        # shared encoder
        self.encoder = network.Encoder()
        self.up = network.ConvBNReLU_Up(128, 128)

        # detail-branch
        self.detail_branch = network.DetailBranch()

        # CA Transformer (fuse detail-branch feature & encoder feature)
        self.fuse_transform = network.TransformerBlock(channels=128, num_heads=4, expansion_factor=2.66)

        # decoder for 2 branches (instance & seed)
        # Decoder(3) for instance branch & Decoder(1) for seed branch
        self.decoders = nn.ModuleList()
        for n in num_classes:
            self.decoders.append(network.Decoder(n))


    def init_output(self, n_sigma=1):
        if sum(self.num_classes) == 4:
            n_sigma = 1  # 1 sigma for circular margin
        else:
            n_sigma = 2  # 2 sigma for elliptical margin

        with torch.no_grad():
            output_conv = self.decoders[0].output_conv
            print('initialize last layer with size: ',
                  output_conv.weight.size())

            output_conv.weight[:, 0:2, :, :].fill_(0)
            output_conv.bias[0:2].fill_(0)

            output_conv.weight[:, 2:2 + n_sigma, :, :].fill_(0)
            output_conv.bias[2:2 + n_sigma].fill_(0.75)

            
    def forward(self, input):

        encoded_feature = self.encoder(input)  # (N, 128, 64, 64)
        encoded_feature = self.up(encoded_feature)

        detail_feature = self.detail_branch(input)  # (N, 128, 128, 128)

        ca_feature_ed, _ = self.fuse_transform(encoded_feature, detail_feature)  # (N, 128, 128, 128)

        # concat (N, 3, h, w) & (N, 1, h, w)
        return torch.cat([decoder.forward(ca_feature_ed) for decoder in self.decoders], 1)