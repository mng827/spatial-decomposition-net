import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvBlock(nn.Module):
    def __init__(self, input_channels, num_conv, num_filters, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        layers = []

        for i in range(num_conv):
            if i == 0:
                in_channels = input_channels
            else:
                in_channels = num_filters[i-1]

            layers.append(nn.Conv2d(in_channels, num_filters[i], kernel_size=kernel_size, stride=stride, padding=padding))
            layers.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*layers)

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
        output = self.net(x)
        return output
