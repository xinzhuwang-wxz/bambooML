import torch
import torch.nn as nn

def get_model(data_config, **kwargs):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
        def forward(self, features):
            return self.fc(features)
    model = M()
    model_info = {
        'input_names': ('features',),
        'input_shapes': {'features': (-1, 2)},
        'output_names': ('output',),
        'dynamic_axes': {'features': {0: 'batch'}, 'output': {0: 'batch'}} 
    }
    return model, model_info

def get_loss(data_config, **kwargs):
    return nn.CrossEntropyLoss()