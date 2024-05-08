import torch
import torch.optim as optim
from .models import *

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
        model = models.resnet18(num_classes = config['num_classes'])
    return model