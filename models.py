import torch
import torch.nn as nn
import torchvision.models as models

class SimpleMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class CNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class DeepCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(DeepCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class RNNModel(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, num_classes=10):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.squeeze(1) if x.shape[1] == 1 else x.mean(dim=1)
        h0 = torch.zeros(1, x.size(0), 128).to(x.device)
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :])

class LSTMModel(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, num_classes=10):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.squeeze(1) if x.shape[1] == 1 else x.mean(dim=1)
        h0 = torch.zeros(1, x.size(0), 128).to(x.device)
        c0 = torch.zeros(1, x.size(0), 128).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

def get_model(model_name, dataset_name, device):
    input_channels = 1 if dataset_name == 'MNIST' else 3
    input_size = 28 * 28 if dataset_name == 'MNIST' else 32 * 32 * 3
    if model_name == 'SimpleMLP':
        return SimpleMLP(input_size, 10).to(device)
    elif model_name == 'CNN':
        return CNN(input_channels, 10).to(device)
    elif model_name == 'DeepCNN':
        return DeepCNN(input_channels, 10).to(device)
    elif model_name == 'ResNet18':
        model = models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 10)
        return model.to(device)
    elif model_name == 'RNN':
        return RNNModel().to(device)
    elif model_name == 'LSTM':
        return LSTMModel().to(device)
    else:
        raise ValueError("Unsupported model.")
