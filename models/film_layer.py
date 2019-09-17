import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FilmLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, num_modulation_factors):
        super(FilmLayer, self).__init__()

        self.output_channels = output_channels
        self.num_modulation_factors = num_modulation_factors

        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.3, inplace=True)

        self.conv2 = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.3, inplace=True)

        self.film_params = nn.Sequential(nn.Linear(num_modulation_factors, 2 * output_channels),
                                         nn.LeakyReLU(negative_slope=0.3, inplace=True),
                                         nn.Linear(2 * output_channels, 2 * output_channels))

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


    def forward(self, x, modality_factor):
        gamma_beta = self.film_params(modality_factor)
        gamma = gamma_beta[:, :self.output_channels]       # (B, C)
        beta = gamma_beta[:, self.output_channels:]        # (B, C)

        output = self.conv1(x)
        output = self.lrelu1(output)

        residual = self.conv2(output)

        gamma = torch.unsqueeze(gamma, dim=-1)
        gamma = torch.unsqueeze(gamma, dim=-1)     # (B, C, H, W)
        beta = torch.unsqueeze(beta, dim=-1)
        beta = torch.unsqueeze(beta, dim=-1)      # (B, C, H, W)

        residual = residual * gamma + beta
        residual = self.lrelu2(residual)

        return output + residual
