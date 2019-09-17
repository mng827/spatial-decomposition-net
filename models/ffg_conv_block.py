import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from .conv_block import ConvBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FFGConvBlock(nn.Module):
    def __init__(self, num_filters, fc_hidden_dim, output_dim, input_image_size, input_channels):
        super(FFGConvBlock, self).__init__()

        self.num_conv = len(num_filters)
        self.num_filters = num_filters
        self.fc_hidden_dim = fc_hidden_dim
        self.output_dim = output_dim
        self.input_image_size = input_image_size
        self.input_channels = input_channels

        self.conv_block = ConvBlock(input_channels=self.input_channels, num_conv=self.num_conv,
                                    num_filters=self.num_filters, kernel_size=3, stride=2, padding=0)

        spatial_dim_output = self.input_image_size
        for _ in range(self.num_conv):
            spatial_dim_output = (spatial_dim_output - 1) // 2

        self.fc = nn.Linear(num_filters[-1] * spatial_dim_output ** 2, self.fc_hidden_dim)

        self.dense_layer_mean = nn.Linear(self.fc_hidden_dim, self.output_dim)
        self.dense_layer_sigma = nn.Linear(self.fc_hidden_dim, self.output_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]
        output = self.conv_block(x)

        output = self.fc(output.view(batch_size, -1))
        mu = self.dense_layer_mean(output)
        log_sigma = self.dense_layer_sigma(output)

        dist = Normal(loc=mu, scale=F.softplus(log_sigma))

        return dist
