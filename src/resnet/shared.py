from enum import Enum

from torch import nn


class ResNet_Backbone(nn.Module):
    pass


class ResNet_Softmax(nn.Module):
    pass


class ResNet_Gaussian(nn.Module):
    pass


class SoftmaxMode(Enum):
    LOGITS = 0
    LOG_SOFTMAX = 1
    SOFTMAX = 2


class GaussianMode(Enum):
    SQR_DISTANCES = 0
    LOG_LIKELIHOODS = 1
    LOG_POSTERIORS = 2
    BOTH = 3
