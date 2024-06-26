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
import scipy.io as sio

def load_data(file_path):
    with h5py.File(file_path, 'r') as f:
        lat = np.array(f['lat'])
        lon = np.array(f['lon'])
        sst_all = np.array(f['sst'])
        time = np.array(f['time'])
    return lat, lon, sst_all, time

def load_mask(file_path):
    data = sio.loadmat(file_path)
    mask = data['sea']
    return mask

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

def build_optimizer(model, initial_lr=0.001, threshold=0.05, patience=3, factor=0.1, min_lr=1e-20):
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=0.0)
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
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import ResNet101_Weights, ResNet50_Weights, ResNet34_Weights, ResNet18_Weights
from torchvision.models import vision_transformer as vit
# import torchvision.transforms.functional as F

class ResizeBlock(nn.Module):
    def __init__(self, input_dim, output_dim=64800):
        super(ResizeBlock, self).__init__()
        # Define a linear layer to transform the input tensor to the desired output size
        self.resize = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        # Apply the linear transformation
        x = self.resize(x)
        return x

class CustomViT(nn.Module):
    def __init__(self, config):
        super(CustomViT, self).__init__()
        self.vit = vit.VisionTransformer(
            image_size=config['image_size'],
            patch_size=config['patch_size'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            hidden_dim=config['hidden_dim'],
            mlp_dim=config['mlp_dim'],
            dropout=0.05,
            num_classes=config['num_classes']  # Set to intermediate dimension
        )
        self.resize_block = ResizeBlock(input_dim=config['num_classes'], output_dim=64800)

    def forward(self, x):
        x = self.vit(x)  # Pass input through Vision Transformer
        x = self.resize_block(x)  # Use the effective resize block
        return x


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

class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        # 第一层卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 第二层卷积层
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # 第三层卷积层
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # 第四层卷积层
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # 自适应平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc = nn.Linear(512, 360*180)  # 假设最终输出的tensor大小为180*360

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.view(-1, 180, 360)  # 调整输出形状与目标一致
        return x

class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        # 第一层卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 第二层卷积层
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # 第三层卷积层
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # 第四层卷积层
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # 第五层卷积层
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        # 第六层卷积层
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        # 第七层卷积层
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 自适应平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc = nn.Linear(512, 360*180)  # 假设最终输出的tensor大小为180*360

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)  # 应用池化减少维度
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)  # 应用池化减少维度

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.maxpool(x)  # 应用池化减少维度

        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x)))
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.view(-1, 64800)  # 调整输出形状与目标一致
        return x

class ResNet(nn.Module):
    def __init__(self, input_shape=(180, 360), n=18):
        super(ResNet, self).__init__()
        # Choose ResNet size
        if n == 18:
            weights = ResNet18_Weights.DEFAULT
            base_model = models.resnet18(weights=weights, num_classes = 64800)
        elif n == 34:
            # weights = ResNet34_Weights.DEFAULT
            # base_model = models.resnet34(weights=weights)
            base_model = models.resnet34()
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
        # model = CustomViT(config)
        model = vit.VisionTransformer(
            image_size=config['image_size'],
            patch_size=config['patch_size'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            hidden_dim=config['hidden_dim'],
            mlp_dim=config['mlp_dim'],
            dropout=0.05,
            num_classes=config['num_classes']
        )
    elif(config['name'] == 'CNN'):
        # model = CNN(
        #     channels=config['channels'],
        #     kernel_size=config['kernel_size'],
        #     padding=config['padding'],
        # )
        model = DeepCNN()
    elif(config['name'] == 'ResNet'):
        # model = ResNet(n=config['size'])
        # weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(num_classes = 64800)
    return model

# save results
import json
import pandas as pd
import datetime

def dump_result(config, data_item, exec_time, mode="train"):
    # Step 1: Generate filename based on the current system time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if (mode == "train"):
        file_name = f"result/result_train_{current_time}.md"
        data_name = f"result/error_train_{current_time}.json"
    elif (mode == "test"):
        file_name = f"result/result_test_{current_time}.md"
        data_name = f"result/error_test_{current_time}.json"
    else:
        pass

    # Step 2: Dump the training data into json file.
    if (mode == "train"):
        all_data = {}
        for i in range(len(data_item)):
            data = data_item[i]
            # Use the iteration number as the key
            all_data[f'iteration_{i}'] = {
                'train_loss': data[0], 
                'train_error': data[1], 
                'validation_loss': data[2], 
                'validation_error': data[3]
            }
            print(f'train_loss = {data[0][-1]}', end=', ')
            print(f'train_error = {data[1][-1]}')
            print(f'validation_loss = {data[2][-1]}', end=', ')
            print(f'validation_error = {data[3][-1]}')
        with open(data_name, 'w') as file:
            json.dump(all_data, file, indent=4)
    elif (mode == "test"):
        all_data = {}
        i = 1
        for sensor_info, test_error in data_item.items():
            all_data[f'test_error_{i}'] = {
                'sensor_num': sensor_info[0],
                'sensor_seed': sensor_info[1],
                'test_error': test_error
            }
            i += 1
        with open(data_name, 'w') as file:
            json.dump(all_data, file, indent=4)

    
    with open(file_name, 'w') as md_file:
        # Step 3: Write config information to the markdown file
        md_file.write("## Description\n")
        md_file.write(f"- time: {current_time}\n\n")
        for category, attributes in config.items():
            md_file.write(f"## {category}\n")
            for attr, value in attributes.items():
                md_file.write(f"- {attr}: {value}\n")
            md_file.write("\n")  # Add an extra newline for better readability
        
        # Step 4: Write training data to the file
        if mode == 'train':
            md_file.write("## Training Data\n")
            md_file.write(f"Training time: {exec_time:.2f} seconds\n")
            for epoch, metrics in data_item.items():
                # Assuming the structure is like [train_loss, train_error, validation_loss, validation_error]
                train_loss, train_error, validation_loss, validation_error = [metrics_list[-1] for metrics_list in metrics]
                md_file.write(f"Epoch {epoch}:\n train_loss={train_loss}\n train_error={train_error}\n validation_loss={validation_loss}\n validation_error={validation_error}\n")
        elif mode == 'test':
            md_file.write("## Testing Data\n")
            md_file.write(f"Testing time: {exec_time:.2f} seconds\n")
            for sensor_info, test_error in data_item.items():
                # Assuming the structure is like test_error
                md_file.write(f"Number of sensor: {sensor_info[0]}; Seed for sensor: {sensor_info[1]}\n")
                md_file.write(f"Testing error: {sum(test_error) / len(test_error)}\n")

# Useless now
def flatten_config(config, parent_key='', sep='_'):
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# Useless now
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
