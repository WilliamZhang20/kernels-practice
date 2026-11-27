import torch


def sample_top_p(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    """Sample one token per row using nucleus (top-p) sampling.

    Args:
        logits: Floating-point tensor with shape [B, V].
        temperature: Positive temperature.
        top_p: Cumulative probability threshold in (0, 1].

    Returns:
        Int64 tensor of sampled token IDs with shape [B].
    """
    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Keep the token that reaches the threshold, as well as all tokens before it.
    remove = cumulative_probs - sorted_probs >= top_p
    filtered_probs = sorted_probs.masked_fill(remove, 0.0)
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

    sampled_sorted_indices = torch.multinomial(filtered_probs, num_samples=1)
    sampled_tokens = torch.gather(sorted_indices, dim=-1, index=sampled_sorted_indices)
    return sampled_tokens.squeeze(-1)


def test_sample_top_p() -> None:
    torch.manual_seed(42)

    # Shape, dtype, and valid token IDs.
    logits = torch.randn(8, 100)
    result = sample_top_p(logits, temperature=1.0, top_p=0.9)
    assert result.shape == (8,)
    assert result.dtype == torch.int64
    assert ((result >= 0) & (result < 100)).all()

    # A top-p threshold below the largest probability keeps only the top token.
    logits = torch.tensor([[10.0, 0.0, -10.0], [0.0, 10.0, -10.0]])
    result = sample_top_p(logits, temperature=1.0, top_p=0.5)
    assert torch.equal(result, torch.tensor([0, 1]))

    # The probabilities are intentionally not in token-ID order. The nucleus is
    # token IDs 1 and 3: 0.5 + 0.3 reaches top_p=0.7. Cumulative sums must be
    # computed from sorted_probs, or token 0 can incorrectly enter the nucleus.
    probs = torch.tensor([[0.15, 0.50, 0.05, 0.30]])
    logits = probs.log()
    samples = torch.cat(
        [sample_top_p(logits, temperature=1.0, top_p=0.7) for _ in range(256)]
    )
    assert torch.isin(samples, torch.tensor([1, 3])).all()
    assert (samples == 1).any()
    assert (samples == 3).any()

    # top_p=1.0 retains the full distribution.
    logits = torch.zeros(1, 4)
    samples = torch.cat(
        [sample_top_p(logits, temperature=1.0, top_p=1.0) for _ in range(256)]
    )
    assert set(samples.tolist()) == {0, 1, 2, 3}

    # Sampling must not modify its input.
    logits = torch.randn(4, 32)
    original = logits.clone()
    sample_top_p(logits, temperature=0.7, top_p=0.9)
    assert torch.equal(logits, original)

    # Temperature changes the distribution when top-p filtering is disabled.
    logits = torch.tensor([[2.0, 1.0]])

    def sample_many(temperature: float) -> torch.Tensor:
        return torch.cat(
            [
                sample_top_p(logits, temperature=temperature, top_p=1.0)
                for _ in range(2000)
            ]
        )

    cold = sample_many(temperature=0.2)
    hot = sample_many(temperature=5.0)
    assert (cold == 0).float().mean() > 0.95
    assert 0.45 < (hot == 0).float().mean() < 0.65


if __name__ == "__main__":
    test_sample_top_p()
    print("All tests passed!")
