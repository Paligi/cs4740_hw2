# AUTO-GENERATED (DO NOT MODIFY)
# NET IDS: RT529,YC2838

import logging
from dataclasses import dataclass
from typing import List

import torch
from torch import nn

from ner.nn.module import Module


class RNN(Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        bias: bool = True,
        nonlinearity: str = "tanh",
    ):
        """
        A multi-layer recurrent neural network with ReLU, tanh, or PReLU nonlinearity to an input sequence.

        Parameters
        ----------
        embedding_dim : int
            Number of dimensions of an input embedding.
        hidden_dim : int
            Number of dimensions of the hidden layer(s).
        output_dim : int
            Number of dimensions for the output layer.
        num_layers : int, default: 1
            Number of layers in the multi-layer RNN model.
        bias : bool, default: True
            If set to False, the input-to-hidden and hidden-to-hidden transformations will not include bias. Note: the
            hidden-to-output transformation remains unaffected by ``bias``.
        nonlinearity : {"tanh", "relu", "prelu"}, default: "tanh"
            Name of the nonlinearity to be applied during the forward pass.
        """
        super().__init__()

        assert num_layers > 0

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        logging.info(f"no shared weights across layers")

        nonlinearity_dict = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "prelu": nn.PReLU()}
        if nonlinearity not in nonlinearity_dict:
            raise ValueError(f"{nonlinearity} not supported, choose one of: [tanh, relu, prelu]")
        self.nonlinear = nonlinearity_dict[nonlinearity]

        # TODO-5-1
        self.input_to_hidden = nn.ModuleList()
        self.hidden_to_hidden = nn.ModuleList()

        for layer_idx in range(num_layers):
            in_dim = embedding_dim if layer_idx == 0 else hidden_dim
            

            self.input_to_hidden.append(
                nn.Linear(in_dim, hidden_dim, bias=bias)
            )
            self.hidden_to_hidden.append(
                nn.Linear(hidden_dim, hidden_dim, bias=bias)
            )

        # hidden->output transform: always has bias (per the docstring)
        self.hidden_to_output = nn.Linear(hidden_dim, output_dim, bias=True)

        # Initialize weights
        self.apply(self.init_weights)

    def _initial_hidden_states(
        self, batch_size: int, init_zeros: bool = False, device: torch.device = torch.device("cpu")
    ) -> List[torch.Tensor]:
        """
        Returns a list of :py:attr:`~ner.nn.models.rnn.RNN.num_layers` number of initial hidden states.

        Parameters
        ----------
        batch_size : int
            The processing batch size.
        init_zeros : bool, default: False
            If False, the hidden states will be initialized using the normal distribution; otherwise, they will be
            initialized as all zeros.
        device: torch.device
            The device to be used in storing the initialized tensors.

        Returns
        -------
        List[torch.Tensor]
            List holding tensors of initialized initial hidden states of shape `(num_layers, batch_size, hidden_dim)`.
        """
        if init_zeros:
            hidden_states = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        else:
            hidden_states = nn.init.xavier_normal_(
                torch.empty(self.num_layers, batch_size, self.hidden_dim, device=device)
            )
        return list(map(torch.squeeze, hidden_states))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        计算输入 embeddings 经过 RNN 的前向传播。

        Parameters
        ----------
        embeddings : torch.Tensor
            输入张量，形状为 `(batch_size, max_length, embedding_dim)`。

        Returns
        -------
        torch.Tensor
            输出张量，形状为 `(batch_size, max_length, output_dim)`。
        """
        batch_size, max_length, _ = embeddings.shape
        device = embeddings.device

        
        hidden_states = self._initial_hidden_states(
            batch_size=batch_size,
            init_zeros=False,    
            device=device
        )

       
        outputs = []
        for t in range(max_length):
            x_t = embeddings[:, t, :]

            
            for layer_idx in range(self.num_layers):
                if layer_idx == 0:
                    input_vec = x_t
                else:
                    input_vec = hidden_states[layer_idx - 1]

              
                h_old = hidden_states[layer_idx]  # (batch_size, hidden_dim)

                h_new = self.input_to_hidden[layer_idx](input_vec) \
                        + self.hidden_to_hidden[layer_idx](h_old)
                h_new = self.nonlinear(h_new)

        
                hidden_states[layer_idx] = h_new

            
            h_top = hidden_states[-1]

            # 3) hidden -> output
            out_t = self.hidden_to_output(h_top)  # (batch_size, output_dim)
            outputs.append(out_t.unsqueeze(1))    # (batch_size, 1, output_dim)

        
        outputs = torch.cat(outputs, dim=1)  # (batch_size, max_length, output_dim)
        return outputs