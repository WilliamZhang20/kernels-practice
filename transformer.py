"""
Very simply:
- Forward pass of a transformer in PyTorch

TODOS:
- Gated Activation Function
- Gated Convolution (recent research in attention replacement)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class KVCache:
    def __init__(self):
        self.k = []
        self.v = []

    def put(self, key, value):
        self.k.append(key)
        self.v.append(value)

    def get(self):
        return self.k, self.v

class RoPE(nn.Module):
    def __init__(self, head_size, block_size):
        super().__init__()
        if head_size % 2 != 0:
            raise ValueError("head_size must be even for RoPE")

        self.head_size = head_size
        pair_idx = torch.arange(head_size // 2, dtype=torch.float32)
        inv_freq = 1.0 / (10000 ** (2 * pair_idx / head_size))
        self.register_buffer("inv_freq", inv_freq)
        
        positions = torch.arange(block_size, dtype=torch.float32)
        angles = torch.einsum("t,d->td", positions, inv_freq)

        sin = angles.sin()
        self.register_buffer("sin", sin)
        
        cos = angles.cos()
        self.register_buffer("cos", cos)

    def _rotate(self, x, cos, sin):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

    def forward(self, q, k, position_offset=0):
        sequence_length = q.shape[-2]
        if k.shape[-2] != sequence_length:
            raise ValueError("q and k must have the same sequence length")
        position_end = position_offset + sequence_length
        if position_offset < 0 or position_end > self.cos.shape[0]:
            raise ValueError("sequence length exceeds RoPE block_size")

        cos = self.cos[position_offset:position_end].to(dtype=q.dtype, device=q.device)
        sin = self.sin[position_offset:position_end].to(dtype=q.dtype, device=q.device)
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)
        
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, head_size, block_size):
        super().__init__()
        self.head_size = head_size

        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
       
        self.rope = RoPE(head_size, block_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape # batch size, sequence, embed dim
        q, k, v = self.query(x), self.key(x), self.value(x)
        
        q, k = self.rope(q, k)

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

def prefill_attn(q, k, v, sm_scale, causal_mask, kv_cache):
    # gonna do a flash attention-like implementation
    kv_cache.put(k, v)
    q_block_size = 64
    kv_block_size = 64
    sequence_length = q.shape[-2]
    
    # B, T, num_heads, head_size
    out = torch.empty_like(q)
    
    for q_start in range(0, sequence_length, q_block_size):
        q_end = min(q_start + q_block_size, sequence_length)
        q_tile = q[..., q_start:q_end, :]

        # State has one value per query row (and batch/head).
        m_running = torch.full(
            q_tile.shape[:-1], float("-inf"), dtype=torch.float32, device=q.device
        )
        l = torch.zeros_like(m_running)
        acc = torch.zeros(
            (*q_tile.shape[:-1], v.shape[-1]), dtype=torch.float32, device=q.device
        )

        for kv_start in range(0, sequence_length, kv_block_size):
            kv_end = min(kv_start + kv_block_size, sequence_length)
            k_tile = k[..., kv_start:kv_end, :]
            v_tile = v[..., kv_start:kv_end, :]
            mask_tile = causal_mask[q_start:q_end, kv_start:kv_end]
        
            # q_tile: (B, heads, q_tile_size, head_size)
            # k_tile: (B, heads, kv_tile_size, head_size)
            # qk:     (B, heads, q_tile_size, kv_tile_size)
            qk = torch.einsum("bhqd,bhkd->bhqk", q_tile, k_tile) * sm_scale
            
            qk = qk.masked_fill(mask_tile, float("-inf"))

            m_new = torch.max(m_running, qk.max(dim=-1).values)

            alpha = torch.exp(m_running - m_new) 
                
            p = torch.exp(qk - m_new.unsqueeze(-1))

            tile_denominator = p.sum(dim=-1) 
            
            l = alpha * l + tile_denominator

            acc = alpha.unsqueeze(-1) * acc + p @ v_tile

            m_running = m_new

        # Normalize by l and write the completed output tile here.
        out_tile = acc / l.unsqueeze(-1)
        out[..., q_start:q_end, :] = out_tile

    return out

def decode_attn(q, k, v, sm_scale, kv_cache, kv_chunk_size=64):
    # Append this step's K/V, then attend over the full accumulated cache.
    kv_cache.put(k, v)
    cached_ks, cached_vs = kv_cache.get()
    k_cache = torch.cat(cached_ks, dim=-2)
    v_cache = torch.cat(cached_vs, dim=-2)

    # k_cache/v_cache: (B, heads, context_length, head_size)
    B, num_heads, context_length, head_size = k_cache.shape
    num_chunks = math.ceil(context_length / kv_chunk_size)
    padded_length = num_chunks * kv_chunk_size
    pad_length = padded_length - context_length

    if pad_length:
        k_cache = F.pad(k_cache, (0, 0, 0, pad_length))
        v_cache = F.pad(v_cache, (0, 0, 0, pad_length))

    # Chunk axis is now explicit: (B, heads, chunks, chunk_size, head_size).
    k_chunks = k_cache.view(B, num_heads, num_chunks, kv_chunk_size, head_size)
    v_chunks = v_cache.view(B, num_heads, num_chunks, kv_chunk_size, head_size)
    valid_keys = (
        torch.arange(padded_length, device=k_cache.device) < context_length
    ).view(num_chunks, kv_chunk_size)
    
    # Each chunk has shape: (B, heads, chunks, chunk_size, head_size)

    # Each chunk starts its own partial online-softmax state.
    query_length = q.shape[-2]
    m_chunks = torch.full(
        (B, num_heads, num_chunks, query_length),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )
    l_chunks = torch.zeros_like(m_chunks)
    acc_chunks = torch.zeros(
        (B, num_heads, num_chunks, query_length, head_size),
        dtype=torch.float32,
        device=q.device,
    )

    # Compute each chunk's partial attention in parallel, then merge its
    scores = torch.einsum("bhqd,bhckd->bhcqk", q, k_chunks) * sm_scale
    scores = scores.masked_fill(~valid_keys[None, None, :, None, :], float("-inf"))

    # softmax statistics and weighted-value accumulator across the chunk axis.
    m_chunks = scores.max(dim=-1).values # shape of (B, heads, chunks, chunk_size)
    p = torch.exp(scores - m_chunks.unsqueeze(-1))
    acc_chunks = torch.einsum("bhcqk,bhckd->bhcqd", p, v_chunks)

    l_chunks = p.sum(dim=-1) # denominator reduction

    m = m_chunks.max(dim=2).values # (B, heads, query_length)
    m_for_l = m.unsqueeze(2) # (B, heads, 1, query_length)
    alpha = torch.exp(m_chunks - m_for_l)
    l = (alpha * l_chunks).sum(dim=2)
    acc = (alpha.unsqueeze(-1) * acc_chunks).sum(dim=2)
    out = acc / l.unsqueeze(-1)

    return out

# New combined attention block (heads in parallel)
class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, head_size, block_size):
        super().__init__()
        if embed_dim % head_size != 0:
            raise ValueError("embed_dim must be divisible by head_size")

        self.num_heads = embed_dim // head_size
        self.head_size = head_size
        self.qkv = nn.Linear(embed_dim, 3 * self.num_heads * head_size)
        self.out = nn.Linear(self.num_heads * head_size, embed_dim)
        self.sm_scale = 1 / math.sqrt(head_size)
        self.rope = RoPE(head_size, block_size)
        # True marks future key positions that causal attention must mask out.
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1),
        )
    
    def forward(self, x, kv_cache=None):
        # x: (B, T, embed_dim); packed qkv: (B, T, 3 * heads * head_size)
        B, T, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Each projection: (B, T, heads * head_size)
        # Attention layout: (B, heads, T, head_size)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        
        # dispatch to prefill/decode attn when necessary
        if kv_cache is None:
            kv_cache = KVCache()

        cache_length = sum(cached_k.shape[-2] for cached_k in kv_cache.k)
        q, k = self.rope(q, k, position_offset=cache_length)

        if not kv_cache.k:
            attn = prefill_attn(
                q,
                k,
                v,
                self.sm_scale,
                self.causal_mask[:T, :T],
                kv_cache,
            )
        else:
            attn = decode_attn(q, k, v, self.sm_scale, kv_cache)
        
        attn = attn.transpose(1, 2).reshape(B, T, -1)
        return self.out(attn), kv_cache

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, token_ids):
        return self.embedding(token_ids)

class PositionalEmbedding(nn.Module):
    def __init__(self, block_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(block_size, embed_dim)

    def forward(self, sequence_length, device):
        positions = torch.arange(sequence_length, device=device)
        return self.embedding(positions)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

class FinalNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(x)


class LMHead(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.proj = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, head_size, block_size, hidden_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadedAttention(embed_dim, head_size, block_size)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffw = FeedForward(embed_dim, hidden_dim)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffw(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        head_size,
        block_size,
        num_layers,
        hidden_dim=None,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * embed_dim

        self.block_size = block_size
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList(
            TransformerBlock(embed_dim, head_size, block_size, hidden_dim)
            for _ in range(num_layers)
        )
        self.final_norm = FinalNorm(embed_dim)
        self.lm_head = LMHead(embed_dim, vocab_size)

    def forward(self, token_ids):
        # token_ids contains integer vocabulary indices and has shape (B, T).
        _, sequence_length = token_ids.shape
        _, sequence_length = token_ids.shape
        _, sequence_length = token_ids.shape
        if sequence_length > self.block_size:
            raise ValueError(
                f"sequence length {sequence_length} exceeds block_size {self.block_size}"
            )

        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))

def smoke_test_transformer_forward():
    vocab_size = 32
    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=16,
        head_size=4,
        block_size=8,
        num_layers=2,
    )
    token_ids = torch.randint(0, vocab_size, (2, 5))
    logits = model(token_ids)

    expected_shape = (2, 5, vocab_size)
    assert logits.shape == expected_shape, (
        f"expected logits shape {expected_shape}, got {tuple(logits.shape)}"
    )
    print(f"Forward pass succeeded: logits shape {tuple(logits.shape)}")


def trace_transformer_shapes():
    vocab_size = 32
    batch_size = 2
    sequence_length = 5
    embed_dim = 16
    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        head_size=4,
        block_size=8,
        num_layers=2,
    )
    token_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
    print(f"token_ids:             {tuple(token_ids.shape)}")

    x = model.token_embedding(token_ids)
    expected = (batch_size, sequence_length, embed_dim)
    assert x.shape == expected
    print(f"token embeddings:      {tuple(x.shape)}")

    for index, block in enumerate(model.blocks):
        x = block(x)
        assert x.shape == expected
        print(f"after block {index}:       {tuple(x.shape)}")

    x = model.final_norm(x)
    assert x.shape == expected
    print(f"after final norm:      {tuple(x.shape)}")

    logits = model.lm_head(x)
    expected_logits = (batch_size, sequence_length, vocab_size)
    assert logits.shape == expected_logits
    print(f"logits:                {tuple(logits.shape)}")


if __name__ == "__main__":
