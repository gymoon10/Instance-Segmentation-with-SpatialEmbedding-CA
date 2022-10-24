import torch
import torch.nn as nn
import models.bisenetv2_custom_wod as network


class BranchedBiSeNetV2_Custom_WOD(nn.Module):
    '''shared encoder + 2 branched decoders'''

    def __init__(self, num_classes, encoder=None):
        super().__init__()

        self.num_classes = num_classes
        print('Creating Branched BiSeNetV2_Custom(Without Detail-Branch) with {} classes'.format(num_classes))

        # shared encoder
        self.encoder = network.Encoder()

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

        # concat (N, 3, h, w) & (N, 1, h, w)
        return torch.cat([decoder.forward(encoded_feature) for decoder in self.decoders], 1)