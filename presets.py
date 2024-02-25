# config_reader
import yaml
from torchvision.models import vision_transformer as vit

def read_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# data_loader
import h5py
import numpy as np

def load_data(file_path):
    with h5py.File(file_path, 'r') as f:
        lat = np.array(f['lat'])
        lon = np.array(f['lon'])
        sst_all = np.array(f['sst'])
        time = np.array(f['time'])
    return lat, lon, sst_all, time

# optimizer
import torch.optim as optim

class CustomLRScheduler:
    def __init__(self, optimizer, threshold, patience, factor=0.1, min_lr=0.00001, history_len=5):
        self.optimizer = optimizer
        self.threshold = threshold  # The range within which we consider the error to be fluctuating
        self.patience = patience  # How many epochs to wait after detecting fluctuation
        self.factor = factor
        self.min_lr = min_lr
        self.history_len = history_len  # How many past epochs to consider for error fluctuation
        self.error_history = []

    def step(self, current_error):
        self.error_history.append(current_error)
        if len(self.error_history) > self.history_len:
            self.error_history.pop(0)

        if len(self.error_history) == self.history_len and max(self.error_history) - min(self.error_history) < self.threshold:
            if self.patience > 0:
                self.patience -= 1
            else:
                self.reduce_lr()

    def reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            new_lr = max(param_group['lr'] * self.factor, self.min_lr)
            param_group['lr'] = new_lr
        print(f"Reducing learning rate to {new_lr}")

def build_optimizer(model, initial_lr=0.001, threshold=0.05, patience=3, factor=0.1, min_lr=0.00001):
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = CustomLRScheduler(optimizer, threshold, patience, factor, min_lr)

    return optimizer, scheduler

# criterion
import torch.nn as nn
class ScaledL2RelativeErrorLoss(nn.Module):
    def __init__(self, epsilon=1e-8, scale_factor=1.0):
        super(ScaledL2RelativeErrorLoss, self).__init__()
        self.epsilon = epsilon
        self.scale_factor = scale_factor

    def forward(self, predictions, targets):
        # Calculate the relative differences
        relative_diffs = (predictions - targets) / (targets + self.epsilon)
        
        # Calculate the L2 norm of the relative differences and scale it
        l2_relative_error = torch.sqrt(torch.sum(torch.pow(relative_diffs, 2))) * self.scale_factor
        
        return l2_relative_error

# model
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet101_Weights, ResNet50_Weights, ResNet34_Weights, ResNet18_Weights
from torchvision.models import vision_transformer as vit

class CNN(nn.Module):
    def __init__(self, channels, kernel_size, padding):
        super(CNN, self).__init__()
        # Define convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=kernel_size, padding=padding),  # Adjusted in_channels to 1
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=kernel_size, padding=padding)  # Final layer with 1 filter
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return x

class ResNet(nn.Module):
    def __init__(self, input_shape=(180, 360), n=18):
        super(ResNet, self).__init__()
        # Choose ResNet size
        if n == 18:
            weights = ResNet18_Weights.DEFAULT
            base_model = models.resnet18(weights=weights)
        elif n == 34:
            weights = ResNet34_Weights.DEFAULT
            base_model = models.resnet34(weights=weights)
        elif n == 50:
            weights = ResNet50_Weights.DEFAULT
            base_model = models.resnet50(weights=weights)
        elif n == 101:
            weights = ResNet101_Weights.DEFAULT
            base_model = models.resnet101(weights=weights)
        else:
            raise ValueError("Unsupported ResNet size. Choose from 18, 34, 50, or 101.")

        self.base_layers = nn.Sequential(*list(base_model.children())[:-2])  # Remove the last two layers
        
        # Assuming input_shape is (height, width), and ResNet outputs (batch_size, channels, height/32, width/32)
        # Calculate output shape after passing through ResNet
        with torch.no_grad():
            self.out_channels = self.base_layers(torch.zeros(1, 3, *input_shape)).shape[1]
            self.out_height = input_shape[0] // 32
            self.out_width = input_shape[1] // 32

        # Upsample to original size
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(self.out_channels, 1, kernel_size=3, stride=32, padding=1, output_padding=1),
            nn.Upsample(size=input_shape, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        x = self.base_layers(x)
        x = self.upsample(x)
        return x

def build_model(config):
    if(config['name'] == 'VisionTransformer'):
        model = vit.VisionTransformer(
            image_size=config['image_size'],
            patch_size=config['patch_size'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            hidden_dim=config['hidden_dim'],
            mlp_dim=config['mlp_dim'],
            num_classes=config['num_classes']
        )
    elif(config['name'] == 'CNN'):
        model = CNN(
            channels=config['channels'],
            kernel_size=config['kernel_size'],
            padding=config['padding'],
        )
    elif(config['name'] == 'ResNet'):
        model = ResNet(n=config['size'])
    return model

# save results
import json
import pandas as pd

def flatten_config(config, parent_key='', sep='_'):
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def save_results(config_file_path, test_results, output_type = 'json', results_output='test_results'):
    results_output_path = f'result/{results_output}.{output_type}'
    # Read YAML configurations
    with open(config_file_path, 'r') as yaml_file:
        configurations = yaml.safe_load(yaml_file)
    flat_config = flatten_config(configurations)
    # Save test results to JSON
    if(output_type == 'json'):
        with open(results_output_path, 'w') as json_file:
            json.dump(flat_config, json_file, indent=4)
            json.dump(test_results, json_file, indent=4)
    elif(output_type == 'csv'):
        config_df = pd.DataFrame(list(flat_config.items()), columns=['Key', 'Value'])
        separator = pd.DataFrame([['---', '---']], columns=['Key', 'Value'])
        results_df = pd.DataFrame(test_results)
        combined_df = pd.concat([config_df, separator, results_df], ignore_index=True)
        combined_df.to_csv(results_output_path, index=False)
    elif(output_type == 'xlsx'):
        with pd.ExcelWriter(results_output_path, engine='openpyxl') as writer:
            config_df = pd.DataFrame(list(flat_config.items()), columns=['Configuration', 'Value'])
            config_df.to_excel(writer, sheet_name='Configurations', index=False)
            results_df = pd.DataFrame(test_results)
            results_df.to_excel(writer, sheet_name='Test Results', index=False)
