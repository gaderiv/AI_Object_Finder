import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class I3D(nn.Module):
    def __init__(self, num_classes=1):
        super(I3D, self).__init__()
        self.resnet3d = models.video.r3d_18(pretrained=True)
        self.resnet3d.fc = nn.Linear(self.resnet3d.fc.in_features, num_classes)

    def forward(self, x):
        # x shape: [batch_size, 3, 16, 224, 224]
        return self.resnet3d(x)