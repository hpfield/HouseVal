import torch
from torch import nn

class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        """
        Initializes the MLPRegressor model.
        
        :param input_size: int, the number of input features
        :param hidden_layers: list of int, each representing the number of neurons in a hidden layer
        :param output_size: int, the number of output features (for regression, typically 1)
        """
        super(MLPRegressor, self).__init__()
        
        # Create the model architecture
        layers = []
        
        # Input layer
        prev_layer_size = input_size
        
        # Hidden layers
        for layer_size in hidden_layers:
            layers.append(nn.Linear(prev_layer_size, layer_size))
            layers.append(nn.ReLU())  # Using ReLU activation function
            prev_layer_size = layer_size
        
        # Output layer
        layers.append(nn.Linear(prev_layer_size, output_size))
        
        # Register the layers
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        :param x: torch.Tensor, input tensor
        :return: torch.Tensor, the model's output
        """
        return self.model(x)
