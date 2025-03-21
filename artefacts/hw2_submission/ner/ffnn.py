# AUTO-GENERATED (DO NOT MODIFY)
# NET IDS: RT529,YC2838

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ner.nn.module import Module


class FFNN(Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1) -> None:
        """
        A multi-layer feed-forward neural network that applies a linear transformation, followed by a ReLU
        nonlinearity, at each layer.

        Parameters
        ----------
        embedding_dim : int
            Number of dimensions of an input embedding.
        hidden_dim : int
            Number of dimensions for the hidden layer(s).
        output_dim : int
            Number of dimensions for the output layer.
        num_layers : int
            Number of hidden layers to initialize.
        """
        super().__init__()

        assert num_layers > 0

        # TODO-4-1
        # Define the layers of the network
        layers = []

    
        layers.append(nn.Linear(embedding_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))

        # Register the layers in the model
        self.network = nn.Sequential(*layers)


        self.apply(self.init_weights)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass through each of the network layers using the given input embeddings.

        Parameters
        ----------
        embeddings : torch.Tensor
            Input tensor of embeddings of shape ``(batch_size, max_length, embedding_dim)``.

        Returns
        -------
        torch.Tensor
            Output tensor resulting from forward pass of shape ``(batch_size, max_length, output_dim)``.
        """
        # TODO-4-2
        batch_size, max_length, _ = embeddings.shape  # Get input shape
        embeddings = embeddings.view(-1, embeddings.shape[-1])
       
        
        output = self.network(embeddings)

        
        output = output.view(batch_size, max_length, -1)

        return output
