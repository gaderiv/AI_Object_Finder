import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientDet3D(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientDet3D, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(1280, num_classes)  

    def forward(self, x):
        # x shape: [batch_size, 3, 16, 224, 224]
        batch_size, channels, frames, height, width = x.shape
        
        # Reshape input to process each frame
        x = x.transpose(1, 2).reshape(batch_size * frames, channels, height, width)
        
        # Extract features for each frame
        features = self.efficientnet.extract_features(x)
        
        # Reshape features back to include temporal dimension
        _, c, h, w = features.shape
        features = features.reshape(batch_size, frames, c, h, w).transpose(1, 2)
        
        # Apply temporal and spatial pooling
        pooled = self.temporal_pool(features)
        
        # Flatten and classify
        flattened = pooled.reshape(batch_size, -1)
        output = self.classifier(flattened)
        
        return output

def get_efficientdet_3d(num_classes=1):
    return EfficientDet3D(num_classes)