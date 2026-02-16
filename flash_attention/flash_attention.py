import torch
import math
import triton
import triton.language as tl

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
        device = Q.device
        dtype = Q.dtype
        batch_size, seq_q, d = Q.shape
        seq_k = K.shape[1]

        # Allocate outputs
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        # Launch Triton backward kernel
        BLOCK_Q = 64
        BLOCK_K = 32
        grid = (triton.cdiv(seq_q, BLOCK_Q), batch_size)

        flash_bwd_kernel[grid](
            Q, K, V, O, L, dO, dQ, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            dO.stride(0), dO.stride(1), dO.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            seq_q, seq_k, ctx.sqrt_d,
            D=d, BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, is_causal=ctx.is_causal,
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


@triton.jit
def flash_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, dO_ptr, dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_dqb, stride_dqq, stride_dqd,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    is_causal: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)

    # Batch base pointers
    Q_ptr = Q_ptr + pid_b * stride_qb
    K_ptr = K_ptr + pid_b * stride_kb
    V_ptr = V_ptr + pid_b * stride_vb
    O_ptr = O_ptr + pid_b * stride_ob
    dO_ptr = dO_ptr + pid_b * stride_dqb
    dQ_ptr = dQ_ptr + pid_b * stride_dqb
    dK_ptr = dK_ptr + pid_b * stride_dkb
    dV_ptr = dV_ptr + pid_b * stride_dvb
    L_ptr = L_ptr + pid_b * stride_lb

    Q_block = tl.make_block_ptr(
        Q_ptr,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(pid_q * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, D),
        order=(1, 0),
    )
    O_block = tl.make_block_ptr(
        O_ptr,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(pid_q * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, D),
        order=(1, 0),
    )
    dO_block = tl.make_block_ptr(
        dO_ptr,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(pid_q * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, D),
        order=(1, 0),
    )
    dQ_block = tl.make_block_ptr(
        dQ_ptr,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(pid_q * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, D),
        order=(1, 0),
    )
    L_block = tl.make_block_ptr(
        L_ptr,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(pid_q * BLOCK_Q,),
        block_shape=(BLOCK_Q,),
        order=(0,),
    )

    Q = tl.load(Q_block)
    O = tl.load(O_block)
    dO = tl.load(dO_block)
    L = tl.load(L_block)

    # local accumulator for dQ
    dQ_local = tl.zeros((BLOCK_Q, D), dtype=tl.float32)

    num_key_blocks = tl.cdiv(N_KEYS, BLOCK_K)
    K_block = tl.make_block_ptr(
        K_ptr,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(BLOCK_K, D),
        order=(1, 0),
    )
    V_block = tl.make_block_ptr(
        V_ptr,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_K, D),
        order=(1, 0),
    )

    for block_idx in range(num_key_blocks):
        K = tl.load(K_block, boundary_check=(0,1), padding_option="zero")
        V = tl.load(V_block, boundary_check=(0,1), padding_option="zero")
        # Recompute scores
        S = tl.dot(Q, tl.trans(K)) / scale
        if is_causal:
            q_idx = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
            k_idx = block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
            mask = k_idx[None, :] > q_idx[:, None]
            S = tl.where(mask, float('-inf'), S)
        P = tl.exp(S - L[:, None])
        # dV contribution: (BLOCK_K, D) = P^T @ dO
        dV_block_local = tl.dot(tl.trans(P), dO)
        # dP = dO @ V^T (BLOCK_Q, BLOCK_K)
        dP = tl.dot(dO, tl.trans(V))
        # D = rowsum(O * dO)
        D_local = tl.sum(O * dO, axis=1)
        # dS = P * (dP - D)
        dS = P * (dP - D_local[:, None])
        # dQ_local += dS @ K / sqrt_d
        dQ_local = dQ_local + tl.dot(dS, K) / scale
        # dK contribution: (BLOCK_K, D) = dS^T @ Q / sqrt_d
        dK_block_local = tl.dot(tl.trans(dS), Q) / scale

        # Atomically add dK_block_local and dV_block_local into global dK/dV
        for kk in range(BLOCK_K):
            key_pos = block_idx * BLOCK_K + kk
            for dd in range(D):
                if key_pos < N_KEYS:
                    tl.atomic_add(dK_ptr + key_pos * stride_dkk + dd * stride_dkd, dK_block_local[kk, dd])
                    tl.atomic_add(dV_ptr + key_pos * stride_dvk + dd * stride_dvd, dV_block_local[kk, dd])

        K_block = K_block.advance((BLOCK_K, 0))
        V_block = V_block.advance((BLOCK_K, 0))

    # store dQ_local to dQ_block
    tl.store(dQ_block, dQ_local)


class FlashAttentionTriton(torch.autograd.Function):
    """Flash Attention using Triton kernel for forward pass."""

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Forward pass using Triton kernel.

        Args:
            Q: Query tensor of shape (batch, seq_q, d)
            K: Key tensor of shape (batch, seq_k, d)
            V: Value tensor of shape (batch, seq_k, d)
            is_causal: Whether to apply causal masking

        Returns:
            Output tensor of shape (batch, seq_q, d)
        """
        # Allocate output tensors O and L
        # Choose block sizes (e.g., BLOCK_Q = BLOCK_K = 64)
        # Configure grid: (num_query_blocks, batch_size)
        # Launch flash_fwd_kernel
        # Save tensors for backward
        # Return O
        device = Q.device
        dtype = Q.dtype
        batch_size, seq_q, d = Q.shape
        seq_k = K.shape[1]
        
        O = torch.empty((batch_size, seq_q, d), device=device, dtype=dtype)
        L = torch.empty((batch_size, seq_q), device=device, dtype=dtype)
        
        # Calculate grid dimensions
        BLOCK_Q = 64
        BLOCK_K = 32
        grid = (triton.cdiv(seq_q, BLOCK_Q), batch_size)

        # Launch Triton kernel for both causal and non-causal cases. Masking is
        # handled inside the kernel via the `is_causal` constexpr parameter.
        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            seq_q, seq_k, math.sqrt(d),
            D=d, BLOCK_Q=64, BLOCK_K=32, is_causal=is_causal
        )

        # Save for backward
        ctx.save_for_backward(Q, K, V, L, O)
        ctx.is_causal = is_causal
        ctx.sqrt_d = math.sqrt(d)
        return O

    @staticmethod
    def backward(ctx, dO):
        """
        Backward pass for FlashAttentionTriton.

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


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    is_causal: tl.constexpr,
):
    """
    Flash Attention forward kernel using online softmax algorithm.

    Each program instance processes one query block for one batch element.
    """
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    Q_ptr = Q_ptr + pid_b * stride_qb
    Q_block = tl.make_block_ptr(
        Q_ptr,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(pid_q * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, D),
        order=(1, 0),
    )
    K_ptr = K_ptr + pid_b * stride_kb
    K_block = tl.make_block_ptr(
        K_ptr,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(BLOCK_K, D),
        order=(1, 0),
    )
    V_ptr = V_ptr + pid_b * stride_vb
    V_block = tl.make_block_ptr(
        V_ptr,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_K, D),
        order=(1, 0),
    )
    O_ptr = O_ptr + pid_b * stride_ob
    O_block = tl.make_block_ptr(
        O_ptr,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(pid_q * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, D),
        order=(1, 0),
    )
    L_ptr = L_ptr + pid_b * stride_lb
    L_block = tl.make_block_ptr(
        L_ptr,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(pid_q * BLOCK_Q,),
        block_shape=(BLOCK_Q,),
        order=(0,),
    )

    Q = tl.load(Q_block)
    num_key_blocks = tl.cdiv(N_KEYS, BLOCK_K)
    m_prev = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    o_prev = tl.zeros((BLOCK_Q, D), dtype=tl.float32)


    for block_idx in range(num_key_blocks):
        K = tl.load(K_block, boundary_check=(0,1), padding_option="zero")
        V = tl.load(V_block, boundary_check=(0,1), padding_option="zero")
        # Compute attention scores
        X = tl.dot(Q, tl.trans(K)) / scale
        if is_causal:
            q_idx = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
            k_idx = block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
            mask = k_idx[None, :] > q_idx[:, None]
            X = tl.where(mask, float('-inf'), X)
        m_i = tl.max(X, axis=-1)
        m = tl.maximum(m_prev, m_i)
        exp_m_prev = tl.exp(m_prev - m)
        exp_x_m = tl.exp(X - m[:,None])
        l = l_prev * exp_m_prev + tl.sum(exp_x_m, axis=-1)
        o = o_prev * exp_m_prev[:,None]
        o = o + tl.dot(exp_x_m, V)
        m_prev = m
        l_prev = l
        o_prev = o

        K_block = K_block.advance((BLOCK_K, 0))
        V_block = V_block.advance((BLOCK_K, 0))

    o_prev = o_prev / l_prev[:,None]
    tl.store(O_block, o_prev)
    tl.store(L_block, m_prev + tl.log(l_prev + 1e-20))