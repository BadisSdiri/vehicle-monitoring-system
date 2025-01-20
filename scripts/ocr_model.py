import torch
import torch.nn as nn

class LicensePlateOCR(nn.Module):
    def __init__(self, num_classes, hidden_dim=256):
        super(LicensePlateOCR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.lstm = nn.LSTM(64 * 32, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        features = self.conv(x)  # Extract spatial features
        features = features.permute(0, 2, 3, 1).flatten(2)  # Reshape for LSTM
        lstm_out, _ = self.lstm(features)
        output = self.fc(lstm_out)
        return output
