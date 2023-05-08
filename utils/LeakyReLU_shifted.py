import torch
import torch.nn as nn


class LeakyReLU_shifted(nn.Module):
    def __init__(self, alpha=0.01, bias=True):
        super(LeakyReLU_shifted, self).__init__()
        self.alpha = alpha
        self.bias = bias
        if self.bias:
            self.bias_param = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if self.bias:
            out = nn.functional.leaky_relu(x, negative_slope=self.alpha) + self.bias_param
        else:
            out = nn.functional.leaky_relu(x, negative_slope=self.alpha)
        return out
