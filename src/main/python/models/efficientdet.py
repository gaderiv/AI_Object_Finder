import torch
from efficientnet_pytorch import EfficientNet

class EfficientDetBackbone(torch.nn.Module):
    def __init__(self, model_name='efficientnet-b0'):
        super(EfficientDetBackbone, self).__init__()
        self.backbone = EfficientNet.from_pretrained(model_name)
        self.pooling = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.backbone.extract_features(x)
        x = self.pooling(x)
        x = x.flatten(start_dim=1)
        return x

class EfficientDet(torch.nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientDet, self).__init__()
        self.backbone = EfficientDetBackbone()
        self.classifier = torch.nn.Linear(self.backbone.backbone._fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x