import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_block import ConvBlock
from .ffg_conv_block import FFGConvBlock
from .decoder import Decoder
from .unet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()
    @staticmethod
    def backward(ctx, grad):
        return grad


class SDNet(nn.Module):
    def __init__(self, input_image_size, num_anatomical_factors, num_modality_factors, num_classes):
        super(SDNet, self).__init__()

        self.input_image_size = input_image_size
        self.num_anatomical_factors = num_anatomical_factors
        self.num_modality_factors = num_modality_factors
        self.num_classes = num_classes

        self.unet = UNet(num_classes=self.num_anatomical_factors, num_levels=5,
                         num_filters=[32, 64, 128, 256, 512], apply_last_layer=True)

        self.modality_encoder = FFGConvBlock(num_filters=[16, 16, 16, 16], fc_hidden_dim=32,
                                             output_dim=self.num_modality_factors,
                                             input_image_size=self.input_image_size,
                                             input_channels=self.num_modality_factors+1)

        self.image_decoder = Decoder(input_channels=self.num_anatomical_factors, hidden_channels=8,
                                     kernel_size=3, stride=1, padding=1,
                                     num_modality_factors=self.num_modality_factors)

        self.segmenter = nn.Sequential(ConvBlock(input_channels=self.num_anatomical_factors,
                                                 num_conv=2, num_filters=[64, 64],
                                                 kernel_size=3, stride=1, padding=1),
                                       nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0))

    def anatomy_encoder(self, x):
        output = self.unet(x)
        output = nn.Softmax(dim=1)(output)
        output = RoundNoGradient.apply(output)
        return output

    def forward(self, x):
        anatomical_factor = self.anatomy_encoder(x)
        modality_factor = self.modality_encoder(torch.cat([x, anatomical_factor], dim=1))

        reconstruction = self.image_decoder(anatomical_factor, modality_factor.rsample())
        segmentation_logits = self.segmenter(anatomical_factor)

        return anatomical_factor, modality_factor, reconstruction, segmentation_logits
