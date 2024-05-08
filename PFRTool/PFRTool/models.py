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