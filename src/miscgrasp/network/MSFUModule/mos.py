import numpy as np
import torch
import torch.nn as nn


class MixtureOfSoftMax(nn.Module):
    def __init__(self, n_components, dim_k, mode,dropout_rate=0.1):
        super().__init__()
        self.temperature = np.sqrt(dim_k)  # Use sqrt for temperature scaling
        self.n_components = n_components
        self.dim_k = dim_k
        self.dropout_rate = dropout_rate
        self.mode = mode
        assert mode in ['dot', 'euc']

        # Dropout and Softmax layers
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax_comp = nn.Softmax(dim=1)
        self.softmax_attn = nn.Softmax(dim=2)

        # Initialize weights for mixture components
        if n_components > 1:
            self.weights = nn.Parameter(torch.empty(n_components, dim_k))  # (n_mix, dim_k)
            nn.init.uniform_(self.weights, -1 / np.sqrt(n_components), 1 / np.sqrt(n_components))  # Efficient initialization

    def forward(self, query, key, value):
        # print(query.shape)
        # print(key.shape)
        # print()
        batch_size, dim_k, seq_len_q = query.shape
        assert dim_k == self.dim_k, f"Dimension mismatch: {dim_k} != {self.dim_k}"

        n_mix = self.n_components
        dim_per_comp = dim_k // n_mix  # Dimension per mixture component

        # Compute mixture weights if there are multiple components
        if n_mix > 1:
            avg_query = query.mean(dim=2, keepdim=True)  # Compute mean over sequence length, (B, dim_k, 1)
            pi = self.softmax_comp(self.weights @ avg_query).view(batch_size * n_mix, 1, 1)

        # Reshape tensors for computation
        query = query.view(batch_size * n_mix, dim_per_comp, seq_len_q).transpose(1, 2)  # (B*n_mix, seq_len_q, dim_per_comp)
        key = key.view(batch_size * n_mix, dim_per_comp, -1)  # (B*n_mix, dim_per_comp, seq_len_k)
        value = value.transpose(1, 2)  # (B, seq_len_k, dim_k)

        # Compute attention scores
        if self.mode == 'dot':
            attn = torch.bmm(query, key) / self.temperature  # (B*n_mix, seq_len_q, seq_len_k)
        elif self.mode == 'euc':
            key = key.transpose(1, 2)
            attn = torch.cdist(query, key, p=2.0) / self.temperature  # (B*n_mix, seq_len_q, seq_len_k)
        attn = self.softmax_attn(attn)
        attn = self.dropout(attn)

        # Apply mixture weighting
        if n_mix > 1:
            attn = (attn * pi).view(batch_size, n_mix, seq_len_q, -1).sum(dim=1)  # Weighted sum across mixtures

        # Compute final output
        output = torch.bmm(attn, value)  # (B, seq_len_q, dim_k)
        return output, attn
