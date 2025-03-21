# AUTO-GENERATED (DO NOT MODIFY)
# NET IDS: RT529,YC2838

from dataclasses import dataclass

import torch
from torch import nn

from ner.nn.module import Module


class TokenEmbedding(Module):
    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx: int = 0):
        """
        Initializes token embeddings (one for each token in the vocabulary) using the given parameters.

        Parameters
        ----------
        vocab_size : int
            The size of the vocabulary; ``vocab_size`` total embeddings are initialized using :py:class:`~nn.Embedding`.
        embedding_dim : int
            The required embedding dimension.
        padding_idx : int
            The token index corresponding to padding tokens; the padded token embedding is a vector of all zeros.
        """
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embedding_dim, 
            padding_idx=padding_idx, 
        )

        self.init_weights(self.embedding)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass to retrieve the token embeddings associated with the input IDs.

        Parameters
        ----------
        input_ids : torch.Tensor
            A (batched) tensor containing input IDs of shape ``(batch_size, max_length)``.

        Returns
        -------
        torch.Tensor
            A tensor of associated embeddings of shape ``(batch_size, max_length, embedding_dim)``.
        """
        return self.embedding(input_ids)