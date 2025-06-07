from torch import nn
import torch
import torchvision.models as models
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()

        base_model = models.resnet18(pretrained = True)

        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512, 512)  # Adjust based on input size (128x128 images)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # x = self.pool(self.relu(self.conv1(x)))
        # x = self.pool(self.relu(self.conv2(x)))
        # x = self.pool(self.relu(self.conv3(x)))
        # x = self.pool(self.relu(self.conv4(x)))
        # x = x.view(x.size(0), -1)  # Flatten
        # x = self.relu(self.fc1(x))
        # x = self.fc2(x)
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        x =  self.relu(self.fc1(x))
        x = self.fc2(x)
        return x