import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientDet(nn.Module):
    def __init__(self, num_classes=2):
        super(EfficientDet, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.model._fc = nn.Linear(self.model._fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
