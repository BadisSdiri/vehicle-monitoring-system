import torch
import torch.nn as nn
from torchvision import models

def get_resnet18(num_classes=5):
    resnet18 = models.resnet18(pretrained=True)
    resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet18 = resnet18.to(device)
    
    return resnet18
