import torch
import torch.nn as nn
import torch.nn.functional as F

from .film_layer import FilmLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Decoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, stride, padding, num_modality_factors):
        super(Decoder, self).__init__()

        self.num_modality_factors = num_modality_factors

        self.film1 = FilmLayer(input_channels, output_channels=hidden_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, num_modulation_factors=self.num_modality_factors)
        self.film2 = FilmLayer(input_channels, output_channels=hidden_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, num_modulation_factors=self.num_modality_factors)
        self.film3 = FilmLayer(input_channels, output_channels=hidden_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, num_modulation_factors=self.num_modality_factors)
        self.film4 = FilmLayer(input_channels, output_channels=hidden_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, num_modulation_factors=self.num_modality_factors)

        self.conv = nn.Conv2d(hidden_channels, 1, kernel_size, stride, padding)
        self.tanh = nn.Tanh()

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

    def forward(self, anatomical_factor, modality_factor):
        output = self.film1(anatomical_factor, modality_factor)
        output = self.film2(output, modality_factor)
        output = self.film3(output, modality_factor)
        output = self.film4(output, modality_factor)

        output = self.conv(output)
        output = self.tanh(output)

        return output
