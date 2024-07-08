import torch
import torch.nn as nn
from torchvision import models

class I3D(nn.Module):
    def __init__(self,num_classes=2):
        super(I3D,self).__init__()
        self.model = models.video.r3d_18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self,x):
        return self.model(x)