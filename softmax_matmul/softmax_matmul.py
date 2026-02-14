import torch
import triton
import triton.language as tl
import torch.nn.functional as F


def softmax_mult(x, V, dim=-1):
    return F.softmax(x, dim=dim) @ V

@triton.jit
def fused_softmax_kernel(
    x_ptr,
    V_ptr,
    output_ptr,
    stride_xbatch,
    stride_xrow,
    stride_xcol,
    stride_Vbatch,
    stride_Vrow,
    stride_Vcol,
    stride_outbatch,
    stride_outrow,
    stride_outcol,
    d1: tl.constexpr,
    d2: tl.constexpr,
    d3: tl.constexpr,
    BLOCK_1: tl.constexpr,
    BLOCK_2: tl.constexpr,
):
    """
    Compute fused softmax(x) @ V using Triton.

    Args:
        x: Input tensor of shape (batch, d1, d2)
        V: Value tensor of shape (batch, d2, d3)
        BLOCK_1: Block size for d1 dimension
        BLOCK_2: Block size for d2 dimension

    Returns:
        Output tensor of shape (batch, d1, d3)
    """
    tl.static_assert(d2 % BLOCK_2 == 0, "d2 must be divisible by BLOCK_2")
    tl.static_assert(d1 % BLOCK_1 == 0, "d1 must be divisible by BLOCK_1")

    pid_batch = tl.program_id(0)
    pid_row = tl.program_id(1)

    x_block = tl.make_block_ptr(
        base=x_ptr + pid_batch*stride_xbatch,
        shape=(d1, d2),
        strides=(stride_xrow, stride_xcol),
        offsets=(pid_row * BLOCK_1, 0),
        block_shape=(BLOCK_1, BLOCK_2),
        order=(1,0)
    )
    
    V_block = tl.make_block_ptr(
        base=V_ptr + pid_batch*stride_Vbatch,
        shape=(d2, d3),
        strides=(stride_Vrow, stride_Vcol),
        offsets=(0, 0),
        block_shape=(BLOCK_2, d3),
        order=(1,0)
    )

    output_block = tl.make_block_ptr(
        base=output_ptr + pid_batch*stride_outbatch,
        shape=(d1, d3),
        strides=(stride_outrow, stride_outcol),
        offsets=(pid_row * BLOCK_1, 0),
        block_shape=(BLOCK_1,d3),
        order=(1,0)
    )

    Num_blocks = tl.cdiv(d2, BLOCK_2)

    m_prev = tl.full((BLOCK_1,), float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros((BLOCK_1,), dtype=tl.float32)
    out_prev = tl.zeros((BLOCK_1, d3), dtype=tl.float32)

    for _ in range(Num_blocks):
        x = tl.load(x_block, boundary_check=(0,1), padding_option="zero")
        v = tl.load(V_block, boundary_check=(0,1), padding_option="zero")

        # Compute block max (Hint: use tl.max and tl.maximum)
        # your code here
        m = tl.maximum(tl.max(x, axis=-1), m_prev)

        # Update running sum with rescaling (Hint: use tl.exp and tl.sum)
        # your code here

        l = l_prev*tl.exp(m_prev-m) + tl.exp(x-m[:,None]).sum(axis=1)
        
        # Scale and accumulate 
        # depending on hardware:
        # on Turing (T4, RTX8000), cast to tl.float16 for tl.dot product only because of this bug: https://github.com/triton-lang/triton/issues/5557
        # on Hopper (H100), no casting necessary
        # your code here
        out = (tl.exp(m_prev-m)*l_prev/l)[:,None]*out_prev + tl.dot((tl.exp(x-m[:,None])/l[:,None]).to(tl.float16), v.to(tl.float16)).to(tl.float32)
        l_prev = l
        out_prev=out
        m_prev=m
        
        x_block = x_block.advance((0, BLOCK_2))
        V_block = V_block.advance((BLOCK_2, 0))

    tl.store(output_block, out_prev)

def fused_softmax(x, V, BLOCK_1=16, BLOCK_2=16):
    """
    Compute fused softmax(x) @ V using Triton.

    Args:
        x: Input tensor of shape (batch, d1, d2)
        V: Value tensor of shape (batch, d2, d3)
        BLOCK_1: Block size for d1 dimension
        BLOCK_2: Block size for d2 dimension

    Returns:
        Output tensor of shape (batch, d1, d3)
    """
    batch_size, d1, d2 = x.shape
    bs, d2 ,d3 = V.shape
    assert batch_size == bs, "Batch size of x and V must match"
    assert d2 == d2, "d2 of x and V must match"
    fused_softmax_output = torch.empty((batch_size, d1, d3), device=x.device, dtype=x.dtype)

    # Calculate grid dimensions
    grid = (batch_size, triton.cdiv(d1, BLOCK_1))

    # Launch kernel
    fused_softmax_kernel[grid](
        x,
        V,
        fused_softmax_output,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        V.stride(0),
        V.stride(1),
        V.stride(2),
        fused_softmax_output.stride(0),
        fused_softmax_output.stride(1),
        fused_softmax_output.stride(2),
        d1,
        d2,
        d3,
        BLOCK_1=BLOCK_1,
        BLOCK_2=BLOCK_2,
    )

    return fused_softmax_output