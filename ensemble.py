from presets import *
from my_utils import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple average
class WeightedAverageEnsemble(nn.Module):
    def __init__(self, models, weights=None):
        """
        Initialize the ensemble model.

        Parameters:
        - models: List of trained PyTorch models.
        - weights: Optional tensor of weights for each model.
        """
        super(WeightedAverageEnsemble, self).__init__()
        self.models = nn.ModuleList(models)

        # Ensure sub-model parameters are not updated during training
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        num_models = len(models)

        # Initialize weights uniformly if not provided
        self.weights = weights if weights is not None else torch.full((num_models,), 1.0 / num_models, dtype=torch.float32, requires_grad=True)
        
        # Ensure weights are treated as parameters of this model
        self.weights = nn.Parameter(self.weights)

    def forward(self, x):
        """
        Forward pass to generate the ensemble prediction.

        Parameters:
        - x: Input tensor of shape (batch_num, 3, 180, 360).

        Returns:
        - Weighted average of model predictions.
        """
        predictions = [model(x) for model in self.models]
        predictions_stack = torch.stack(predictions)
        
        # Calculate weighted average, using weights reshaped for broadcasting
        ensemble_prediction = torch.sum(predictions_stack * self.weights.view(-1, 1, 1), dim=0)
        
        return ensemble_prediction

# Bagging, use it directly
class BaggingEnsemble(nn.Module):
    def __init__(self, models):
        """
        Initialize the bagging ensemble model.

        Parameters:
        - models: List of untrained PyTorch models.
        """
        super(BaggingEnsemble, self).__init__()
        self.models = nn.ModuleList(models)

        # Set model parameters to not require gradients, as they won't be trained
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Forward pass to generate the ensemble prediction by averaging.

        Parameters:
        - x: Input tensor of shape (batch_num, 3, 180, 360).

        Returns:
        - Ensemble prediction as the average of model predictions.
        """
        # Collect predictions from each model
        predictions = [model(x) for model in self.models]

        # Stack and average predictions
        predictions_stack = torch.stack(predictions)
        ensemble_prediction = torch.mean(predictions_stack, dim=0)
        
        return ensemble_prediction

# Stacking
class StackingEnsemble(nn.Module):
    def __init__(self, models, meta_model):
        """
        Initialize the stacking ensemble model.

        Parameters:
        - models: List of trained PyTorch models (base models).
        - meta_model: A PyTorch model that will learn to combine the predictions (meta-model).
        """
        super(StackingEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False  # Freeze base models

        self.meta_model = meta_model

    def forward(self, x):
        """
        Forward pass to generate the ensemble prediction.

        Parameters:
        - x: Input tensor of shape (batch_size, 3, 180, 360).

        Returns:
        - Final ensemble prediction from the meta-model.
        """
        # Generate base model predictions with an added channel dimension
        base_preds = [torch.unsqueeze(model(x).view(-1,180,360), dim=1) for model in self.models]  # Each tensor shape: (batch_size, 1, 180, 360)
        base_preds_stack = torch.cat(base_preds, dim=1)  # Shape: (batch_size, num_models, 180, 360)

        # Meta-model makes final prediction
        ensemble_prediction = self.meta_model(base_preds_stack)
        
        return ensemble_prediction

class FCNMetaModel(nn.Module):
    def __init__(self, model_num):
        """
        Initialize a fully connected neural network (FCNN) based meta-model.
        
        Parameters:
        - model_num: The number of models being ensembled.
        """
        super(FCNMetaModel, self).__init__()
        input_dim = model_num * 180 * 360  # Flattened size of input
        
        # Define the fully connected layers
        self.fc = nn.Linear(input_dim, 512)  # Example dimension reduction
        self.fc2 = nn.Linear(512, 256)        # Further reduction
        self.fc3 = nn.Linear(256, 180*360)    # Output layer to match the target size

    def forward(self, x):
        """
        Forward pass of the FCN Meta-Model.

        Parameters:
        - x: Input tensor of shape (batch_size, model_num, 180, 360).

        Returns:
        - Tensor of shape (batch_size, 180, 360) as the final ensemble prediction.
        """
        # Flatten the input
        x = x.view(x.size(0), -1)
        
        # Pass through the fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Reshape to the desired output shape
        x = x.view(-1, 180, 360)
        return x

class CNNMetaModel(nn.Module):
    def __init__(self, model_num):
        """
        Initialize a CNN-based meta-model for stacking ensemble.

        Parameters:
        - model_num: The number of models being ensembled, which corresponds to the number of channels in the input tensor.
        """
        super(CNNMetaModel, self).__init__()
        self.conv1 = nn.Conv2d(model_num, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=1)  # Reduce to single channel output
        
        # Additional layers can be added based on the specific requirements of your task.
        # For instance, batch normalization, dropout, more conv layers, etc.

    def forward(self, x):
        """
        Forward pass of the CNNMetaModel.

        Parameters:
        - x: Input tensor of shape (batch_size, model_num, 180, 360), where model_num is the number of stacked model predictions.

        Returns:
        - Tensor of shape (batch_size, 180, 360) as the final ensemble prediction.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.squeeze(x, dim=1)  # Remove the channel dimension to match desired output shape
        return x

class UNetMetaModel(nn.Module):
    def __init__(self, model_num):
        """
        Initialize a U-Net-based meta-model.

        Parameters:
        - model_num: The number of models being ensembled, determining the number of input channels.
        """
        super(UNetMetaModel, self).__init__()
        self.encoder1 = nn.Conv2d(model_num, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Forward pass of the U-Net Meta-Model.

        Parameters:
        - x: Input tensor of shape (batch_size, model_num, 180, 360).

        Returns:
        - Tensor of shape (batch_size, 180, 360) as the final ensemble prediction.
        """
        # Encoder
        e1 = F.relu(self.encoder1(x))
        p1 = self.pool1(e1)
        e2 = F.relu(self.encoder2(p1))
        p2 = self.pool2(e2)
        
        # Bottleneck
        b = F.relu(self.bottleneck(p2))
        
        # Decoder
        d2 = F.relu(self.decoder2(self.upconv2(b)))
        d1 = self.decoder1(self.upconv1(d2))
        
        return torch.squeeze(d1, dim=1)  # Remove channel dimension

# Feature Fusion
class FeatureFusionModel(nn.Module):
    def __init__(self, models, feature_size):
        """
        Initialize the Feature Fusion ensemble model.
        
        Parameters:
        - models: List of models to be ensembled.
        - feature_size: Size of the transformed feature vector for each model.
        """
        super(FeatureFusionModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.feature_transforms = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((feature_size, feature_size)),
            nn.Flatten(),
            nn.Linear(feature_size*feature_size*16, feature_size)
        ) for _ in models])

        # Fusion and prediction layers
        self.fusion_layer = nn.Linear(len(models)*feature_size, feature_size)
        self.prediction_layer = nn.Linear(feature_size, 180*360)  # Adjust based on the final output size

    def forward(self, x):
        # Generate features from each model and transform them
        features = [transform(torch.unsqueeze(model(x).view(-1,180,360), dim=1)) for model, transform in zip(self.models, self.feature_transforms)]
        
        # Concatenate all features
        fused_features = torch.cat(features, dim=1)
        
        # Pass through fusion and prediction layers
        x = F.relu(self.fusion_layer(fused_features))
        x = self.prediction_layer(x)
        
        # Reshape to the target output shape
        x = x.view(-1, 180, 360)
        return x


def build_ensemble_model(config):
    method = config['model']['method']
    models = []
    for i in range(config['model']['num_models']):
        model = build_model(config[f'model_{i+1}'])
        if method != "feature_fusion":
            model.load_state_dict(torch.load(config[f'model_{i+1}']['path']))
        models.append(model)
    if method == "blending":
        return WeightedAverageEnsemble(models)
    elif method == "stacking":
        return StackingEnsemble(models, FCNMetaModel(config['model']['num_models']))
    elif method == "bagging":
        return BaggingEnsemble(models)
    elif method == "feature_fusion":
        return FeatureFusionModel(models, feature_size=config['model']['feature_size'])