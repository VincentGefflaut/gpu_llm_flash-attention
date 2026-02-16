import torch
import math

class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_causal=False
        ):
        """
        Flash Attention forward pass using tiled online softmax.

        Args:
            ctx: Context object for saving tensors for backward
            Q: Query tensor of shape (batch, seq_len, head_dim)
            K: Key tensor of shape (batch, seq_len, head_dim)
            V: Value tensor of shape (batch, seq_len, head_dim)
            is_causal: Whether to apply causal masking (default: False)

        Returns:
            O: Output tensor of shape (batch, seq_len, head_dim)
        """
        # Your code here

        # Use tiled computation with online softmax:
        # - Outer loop over query blocks
        # - Inner loop over key/value blocks
        # - Maintain running (m, l, O) accumulators
        # - Handle causal masking when is_causal=True

        # IMPORTANT: Save tensors needed for backward
        # ctx.save_for_backward(Q, K, V, L, O)
        # ctx.is_causal = is_causal
        # ctx.sqrt_d = sqrt_d

        block_size = 16
        *batch_size, seq_len, head_dim = Q.shape
        sqrt_d = math.sqrt(head_dim)
        device = Q.device
        O = torch.zeros((*batch_size, seq_len, head_dim), device=device)
        L = torch.zeros((*batch_size, seq_len), device=device)
        for j in range(0, seq_len, block_size):
            block_end = min(j + block_size, seq_len)
            Q_block = Q[:, j:block_end, :]  # (batch, block_q, head_dim)
            block_q = block_end - j
            m_prev = torch.full((*batch_size, block_q), float("-inf"), device=device)
            l_prev = torch.zeros((*batch_size, block_q), device=device)
            o_prev = torch.zeros((*batch_size, block_q, head_dim), device=device)
            for i in range(0, seq_len, block_size):
                inner_block_end = min(i + block_size, seq_len)
                K_block = K[:, i:inner_block_end, :]  # (batch, block_k, head_dim)
                V_block = V[:, i:inner_block_end, :]  # (batch, block_k, head_dim)
                # Compute attention scores
                X = torch.matmul(Q_block, K_block.transpose(-2, -1)) / sqrt_d  # (batch, block_q, block_k)
                # Causal masking
                if is_causal:
                    q_idx = torch.arange(j, block_end, device=device).unsqueeze(-1)
                    k_idx = torch.arange(i, inner_block_end, device=device).unsqueeze(0)
                    mask = (q_idx < k_idx).unsqueeze(0)  # (1, block_q, block_k)
                    X = X.masked_fill(mask, float('-inf'))
                m_i = X.max(dim=-1).values  # (batch, block_q)
                m = torch.maximum(m_prev, m_i)
                exp_m_prev = torch.exp(m_prev - m)
                exp_x_m = torch.exp(X - m.unsqueeze(-1))
                l = l_prev * exp_m_prev + exp_x_m.sum(dim=-1)
                o = o_prev * exp_m_prev.unsqueeze(-1)
                o = o + torch.matmul(exp_x_m, V_block)
                m_prev = m
                l_prev = l
                o_prev = o
            o_prev = o_prev / l_prev.unsqueeze(-1)
            O[:, j:block_end, :] = o_prev
            # Save log-sum-exp per query: m + log(l)
            L[:, j:block_end] = m_prev + torch.log(l_prev + 1e-20)
        # Save for backward
        ctx.save_for_backward(Q, K, V, L, O)
        ctx.is_causal = is_causal
        ctx.sqrt_d = sqrt_d
        return O


    @staticmethod
    def backward(ctx, dO):
        """
        Attention backward pass.

        Args:
            ctx: Context object with saved tensors
            dO: Gradient of loss w.r.t. output

        Returns:
            dQ: Gradient w.r.t. Q
            dK: Gradient w.r.t. K
            dV: Gradient w.r.t. V
            None: No gradient for is_causal
        """
        Q, K, V, L, O = ctx.saved_tensors
        dQ, dK, dV, _ = attention_backward_impl(
            Q, K, V, L, O, dO, ctx.sqrt_d, ctx.is_causal
        )
        return dQ, dK, dV, None

def attention_backward_impl(Q, K, V, L, O, dO, sqrt_d, is_causal):
    """
    Backward pass implementation for Flash Attention.

    Uses standard attention gradient formulas with recomputation of P
    from saved log-sum-exp values L.

    Args:
        Q: Query tensor of shape (batch, seq_q, d)
        K: Key tensor of shape (batch, seq_k, d)
        V: Value tensor of shape (batch, seq_k, d)
        L: Log-sum-exp values from forward pass, shape (batch, seq_q)
        O: Output from forward pass, shape (batch, seq_q, d)
        dO: Gradient of loss w.r.t. output, shape (batch, seq_q, d)
        sqrt_d: Square root of head dimension (for scaling)
        is_causal: Whether to apply causal masking

    Returns:
        dQ: Gradient w.r.t. Q
        dK: Gradient w.r.t. K
        dV: Gradient w.r.t. V
        None: Placeholder for compatibility
    """
    # Your code here
    #
    # Steps:
    # 1. Compute D = rowsum(O ⊙ dO)
    # 2. Recompute S = Q @ K^T / sqrt(d)
    # 3. Apply causal mask if is_causal
    # 4. Recompute P = exp(S - L)
    # 5. Compute dV = P^T @ dO
    # 6. Compute dP = dO @ V^T
    # 7. Compute dS = P ⊙ (dP - D)
    # 8. Compute dQ = dS @ K / sqrt(d)
    # 9. Compute dK = dS^T @ Q / sqrt(d)
    device = Q.device
    seq_len = Q.shape[1]
    D = (dO * O).sum(dim=-1, keepdim=True)
    S = torch.matmul(Q, K.transpose(-2, -1)) / sqrt_d # (batch_size, seq_len, seq_len)
    if is_causal:
        q_idx = torch.arange(0, seq_len, device=device).unsqueeze(-1)
        k_idx = torch.arange(0, seq_len, device=device).unsqueeze(0)
        mask = (q_idx < k_idx).unsqueeze(0)  # (1, block_q, block_k)
        S = S.masked_fill(mask, float('-inf'))
    P = torch.exp(S - L.unsqueeze(-1))
    dV = P.transpose(-2, -1) @ dO
    dP = dO @ V.transpose(-2, -1)
    dS = P * (dP - D)
    dQ = dS @ K / sqrt_d
    dK = dS.transpose(-2, -1) @ Q / sqrt_d
    return dQ, dK, dV, None