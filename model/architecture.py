import torch.nn as nn
import torch
import torchvision.models as models

class LocationCNN(nn.Module):
    def __init__(self, dropout_rate=0.3, weights=True):
        super(LocationCNN, self).__init__()

        self.backbone = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=dropout_rate),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout_rate),

            nn.Linear(256, 2)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _freeze_early_layers(self):
        layers_to_freeze = 6
        for i, child in enumerate(self.backbone.children()):
            if i < layers_to_freeze:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        coordinates = self.classifier(x)
        return coordinates

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True