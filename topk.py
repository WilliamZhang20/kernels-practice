

import torch



def slow_moe_router(

    logits: torch.Tensor,

    k: int,

):

    """

    Args:

        logits: [N, E]

            N = number of tokens

            E = number of experts

        k: number of experts selected per token



    Returns:

        expert_indices: [N, k], int64

        expert_weights: [N, k], same dtype as logits

    """

    N, E = logits.shape



    expert_indices = torch.empty(

        (N, k), device=logits.device, dtype=torch.long

    )

    expert_weights = torch.empty(

        (N, k), device=logits.device, dtype=logits.dtype

    )



    # Compute softmax across ALL experts

    probs = torch.softmax(logits, dim=-1)

    # Select top-k experts 
    
    selected_probs, selected = torch.topk(probs, k, dim=-1)

    expert_indices = selected

    expert_weights = selected_probs /selected_probs.sum(dim=-1, keepdim=True)



    return expert_indices, expert_weights



logits = torch.tensor([[1.0, 3.0, 2.0, 0.0]])

k = 2

expert_indices, expert_weights = slow_moe_router(logits, k)



print(expert_indices)

print(expert_weights)
