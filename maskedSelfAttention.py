import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, head_size, block_size):
        super().__init__()
        self.head_size = head_size

        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape # batch size, sequence, embed dim
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = q @ k.transpose(-2, -1) / self.head_size ** 0.5
        # mask BEFORE softmax so softmax doesn't count lookaheads in normaliz
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        # softmax
        normalized = F.softmax(scores, dim=-1)
        return normalized @ v

class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim, head_size, block_size):
        super().__init__()
        if embed_dim % head_size != 0:
            raise ValueError("embed_dim must be divisible by head_size")

        self.num_heads = embed_dim // head_size
        self.heads = nn.ModuleList(
            SelfAttention(embed_dim, head_size, block_size)
            for _ in range(self.num_heads)
        )
        self.proj = nn.Linear(self.num_heads * head_size, embed_dim)

    def forward(self, x):
        concatenated = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.proj(concatenated)
