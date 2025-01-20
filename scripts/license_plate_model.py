import torch
import torch.nn as nn
import torchvision.models as models

class LicensePlateOCR(nn.Module):
    def __init__(self, num_classes):
        super(LicensePlateOCR, self).__init__()
        
        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Identity()  

        self.rnn = nn.LSTM(
            input_size=512,  
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Linear(256 * 2, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        features = self.feature_extractor(x)  
        features = features.unsqueeze(1)  
        rnn_out, _ = self.rnn(features)  
        output = self.classifier(rnn_out[:, -1, :])  

        return output
def get_license_plate_ocr_model(num_classes):
    model = LicensePlateOCR(num_classes=num_classes)
    return model
