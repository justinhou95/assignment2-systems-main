import torch
from einops import rearrange, einsum
import math


import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    q = tl.load(
        Q_block_ptr, boundary_check=(0, 1), padding_option="zero"
    )  # (Q_TILE_SIZE, D)

    m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    q_start = query_tile_index * Q_TILE_SIZE
    # Causal: only process key tiles up to and including the current query tile
    n_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    if is_causal:
        n_k_tiles = tl.minimum(n_k_tiles, query_tile_index + 1)

    for key_tile_index in range(n_k_tiles):
        k = tl.load(
            K_block_ptr, boundary_check=(0, 1), padding_option="zero"
        )  # (K_TILE_SIZE, D)
        v = tl.load(
            V_block_ptr, boundary_check=(0, 1), padding_option="zero"
        )  # (K_TILE_SIZE, D)

        s = tl.dot(q, tl.trans(k)) * scale  # (Q_TILE_SIZE, K_TILE_SIZE)

        if is_causal:
            # Build per-element causal mask: mask out positions where k_idx > q_idx
            q_idx = q_start + tl.arange(0, Q_TILE_SIZE)  # (Q_TILE_SIZE,)
            k_idx = key_tile_index * K_TILE_SIZE + tl.arange(
                0, K_TILE_SIZE
            )  # (K_TILE_SIZE,)
            causal_mask = k_idx[None, :] > q_idx[:, None]  # (Q_TILE_SIZE, K_TILE_SIZE)
            s = tl.where(causal_mask, float("-inf"), s)

        m_new = tl.maximum(m, tl.max(s, axis=1))  # (Q_TILE_SIZE,)
        p = tl.exp(s - m_new[:, None])  # (Q_TILE_SIZE, K_TILE_SIZE)
        l = tl.exp(m - m_new) * l + tl.sum(p, axis=1)  # (Q_TILE_SIZE,)
        o = tl.exp(m - m_new)[:, None] * o + tl.dot(p, v)  # (Q_TILE_SIZE, D)
        m = m_new

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    o = o / l[:, None]
    L = m + tl.log(l)  # log-sum-exp, needed for backward

    tl.store(O_block_ptr, o, boundary_check=(0, 1))
    tl.store(L_block_ptr, L, boundary_check=(0,))


class FlashattentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=True):
        output_shape = Q.shape
        Q = rearrange(Q, "... s d -> (...) s d").contiguous()
        K = rearrange(K, "... s d -> (...) s d").contiguous()
        V = rearrange(V, "... s d -> (...) s d").contiguous()
        n_row, n_seq, d_k = Q.shape

        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16
        D = triton.next_power_of_2(d_k)
        scale = 1.0 / math.sqrt(d_k)

        O = torch.empty_like(Q)
        L = torch.empty((n_row, n_seq), device=Q.device, dtype=torch.float32)

        grid = (triton.cdiv(n_seq, Q_TILE_SIZE), n_row)
        flash_fwd_kernel[grid](
            Q,
            K,
            V,
            O,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            n_seq,
            n_seq,
            scale,
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )

        ctx.save_for_backward(Q, K, V, O, L)

        return O.view(output_shape)

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


class FlashattentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        q_tile: int = 64
        k_tile: int = 64
        n_batch, n_heads, n_seq, d_k = Q.shape
        scale = 1.0 / math.sqrt(d_k)

        # Flatten batch and heads for simpler indexing
        Q = rearrange(Q, "b h s d -> (b h) s d")
        K = rearrange(K, "b h s d -> (b h) s d")
        V = rearrange(V, "b h s d -> (b h) s d")
        B = Q.shape[0]

        O = torch.zeros_like(Q)
        # Running log-sum-exp per query (for compatibility with Triton version)
        L = torch.full((B, n_seq), float("-inf"), device=Q.device, dtype=torch.float32)

        n_q_tiles = math.ceil(n_seq / q_tile)
        n_k_tiles = math.ceil(n_seq / k_tile)

        for qi in range(n_q_tiles):
            q_start = qi * q_tile
            q_end = min(q_start + q_tile, n_seq)
            q = Q[:, q_start:q_end, :].float()  # (B, Tq, d)

            m = torch.full(
                (B, q_end - q_start),
                float("-inf"),
                device=Q.device,
                dtype=torch.float32,
            )
            l = torch.zeros((B, q_end - q_start), device=Q.device, dtype=torch.float32)
            o = torch.zeros(
                (B, q_end - q_start, d_k), device=Q.device, dtype=torch.float32
            )

            # Causal: skip key tiles that are entirely after the current query tile
            ki_limit = (qi + 1) if is_causal else n_k_tiles
            for ki in range(min(ki_limit, n_k_tiles)):
                k_start = ki * k_tile
                k_end = min(k_start + k_tile, n_seq)
                k = K[:, k_start:k_end, :].float()  # (B, Tk, d)
                v = V[:, k_start:k_end, :].float()  # (B, Tk, d)

                # (B, Tq, Tk)
                s = torch.bmm(q, k.transpose(-2, -1)) * scale

                if is_causal:
                    q_idx = torch.arange(q_start, q_end, device=Q.device).unsqueeze(
                        1
                    )  # (Tq, 1)
                    k_idx = torch.arange(k_start, k_end, device=Q.device).unsqueeze(
                        0
                    )  # (1, Tk)
                    s = s.masked_fill(k_idx > q_idx, float("-inf"))

                m_new = torch.maximum(m, s.max(dim=-1).values)  # (B, Tq)
                p = torch.exp(s - m_new.unsqueeze(-1))  # (B, Tq, Tk)
                l = torch.exp(m - m_new) * l + p.sum(dim=-1)  # (B, Tq)
                o = torch.exp(m - m_new).unsqueeze(-1) * o + torch.bmm(
                    p, v
                )  # (B, Tq, d)
                m = m_new

            o = o / l.unsqueeze(-1)
            O[:, q_start:q_end, :] = o.to(Q.dtype)
            L[:, q_start:q_end] = m + torch.log(l)

        O = rearrange(O, "(b h) s d -> b h s d", b=n_batch, h=n_heads)
        return O

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


if __name__ == "__main__":
    num_heads = 1
    d_k = 16
    n_batch = 256
    n_seq = 100

    Q = torch.zeros(size=[n_batch, num_heads, n_seq, d_k], device="cuda")
    K = torch.zeros(size=[n_batch, num_heads, n_seq, d_k], device="cuda")
    V = torch.zeros(size=[n_batch, num_heads, n_seq, d_k], device="cuda")

    output = FlashattentionTriton.apply(Q, K, V, True)
    print(output)
