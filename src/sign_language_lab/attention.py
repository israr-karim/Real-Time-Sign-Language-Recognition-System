"""Shared attention and positional-encoding definitions from the original notebooks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HandUpperBodyAttention(nn.Module):
    def __init__(self, input_dim, num_patches_h=14, num_patches_w=14):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
        )

        # Calculate actual number of patches from ViViT output
        # ViViT uses 16x16 patches for 224x224 images, resulting in 14x14 patches
        total_patches = (224 // 16) * (224 // 16)  # Should be 196

        # Initialize position bias
        pos_bias = torch.zeros(total_patches)
        # Add more weight to upper body region (top 60% of the frame)
        upper_region = int(0.6 * total_patches)
        pos_bias[:upper_region] = 1.0
        self.position_bias = nn.Parameter(pos_bias)

    def forward(self, x):
        # x: [batch_size, num_frames, num_patches, feature_dim]
        batch_size, num_frames, num_patches, feature_dim = x.shape

        # Compute attention scores
        attention_weights = self.attention(
            x
        )  # [batch_size, num_frames, num_patches, 1]

        # Add position bias (broadcasting to match dimensions)
        position_bias = self.position_bias[:num_patches].view(1, 1, -1, 1)
        attention_weights = attention_weights + position_bias

        # Apply softmax
        attention_weights = F.softmax(attention_weights, dim=2)

        # Weight the input features
        weighted_features = x * attention_weights

        # Average over patches
        return weighted_features.sum(dim=2)


class SpatialAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [batch_size, num_frames, num_patches, feature_dim]
        query = self.query(x)  # [batch_size, num_frames, num_patches, feature_dim]
        key = self.key(x).transpose(
            -2, -1
        )  # [batch_size, num_frames, feature_dim, num_patches]
        value = self.value(x)  # [batch_size, num_frames, num_patches, feature_dim]

        # Compute attention scores
        attention_scores = torch.matmul(query, key) / (
            x.shape[-1] ** 0.5
        )  # Scaled dot product
        attention_weights = self.softmax(
            attention_scores
        )  # [batch_size, num_frames, num_patches, num_patches]

        # Aggregate patches
        aggregated_patches = torch.matmul(
            attention_weights, value
        )  # [batch_size, num_frames, num_patches, feature_dim]
        return aggregated_patches.mean(dim=2)


class PositionalEncoding(nn.Module):
    def __init__(self, feature_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, feature_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, feature_dim, 2)
            * (-torch.log(torch.tensor(10000.0)) / feature_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [batch_size, seq_len, feature_dim]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class EnhancedPositionalEncoding(nn.Module):
    def __init__(self, feature_dim, max_len=5000, dropout=0.1):
        super().__init__()
        pe = torch.zeros(max_len, feature_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, feature_dim, 2)
            * (-torch.log(torch.tensor(10000.0)) / feature_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)

        # Add learnable parameters for gesture-specific timing
        self.gesture_timing = nn.Parameter(torch.randn(1, max_len, feature_dim) * 0.02)

    def forward(self, x):
        seq_len = x.size(1)
        pos_encoding = self.pe[:, :seq_len] + self.gesture_timing[:, :seq_len]
        return self.dropout(x + pos_encoding)


class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        attended, _ = self.attention(x, x, x)
        return self.norm(x + attended)
