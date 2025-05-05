import torch
import torch.nn as nn


class LSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return out
    
class Embedder(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers):
        super(Embedder, self).__init__()
        self.embedder = LSTMBlock(feature_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        x = self.embedder(x)
        return torch.relu(self.fc(x))
    
class Recovery(nn.Module):
    def __init__(self, hidden_dim, feature_dim, num_layers):
        super(Recovery, self).__init__()
        self.recovery = LSTMBlock(hidden_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, feature_dim)
        
    def forward(self, x):
        x = self.recovery(x)
        return torch.sigmoid(self.fc(x))
    
class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, num_layers):
        super(Generator, self).__init__()
        self.generator = LSTMBlock(noise_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        x = self.generator(x)
        return torch.relu(self.fc(x))
    
    
class Supervisor(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(Supervisor, self).__init__()
        self.supervisor = LSTMBlock(hidden_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        x = self.supervisor(x)
        return torch.relu(self.fc(x))
    
    
class Discriminator(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(Discriminator, self).__init__()
        self.discriminator = LSTMBlock(hidden_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.discriminator(x)
        return torch.sigmoid(self.fc(x))
    
    
    
    
class TimeGAN(nn.Module):
    def __init__(self, feature_dim, hidden_dim=24, num_layers=3, noise_dim=32):
        super(TimeGAN, self).__init__()
        
        self.embedder = Embedder(feature_dim, hidden_dim, num_layers)
        self.recovery = Recovery(hidden_dim, feature_dim, num_layers)
        self.generator = Generator(noise_dim, hidden_dim, num_layers)
        self.supervisor = Supervisor(hidden_dim, num_layers)
        self.discriminator = Discriminator(hidden_dim, num_layers)
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError("TimeGAN utilise un entraînement étapre par étape, pas un seul forward global")
        
        