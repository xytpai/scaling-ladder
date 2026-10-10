# Attention 中的 MHA、GQA、Flash Attention 与 MLA

自注意力：每个 token 用自己的 Q 和全部 K 做点积，softmax 后去加权 V。序列长度 S、头宽 d 时，分数矩阵是 **S×S**，缩放是 `1/sqrt(d)`。

```text
scores = Q K^T / sqrt(d)
out    = softmax(scores) V
```

Q、K、V 都从同一条隐状态 h（`[B, S, D]`）投影出来。头数 H，一头内容宽 d。下面四节按这个顺序：普通 MHA 用 PyTorch 把 S×S 直接拼出来，GQA 让多头共用一份 K/V，Flash Attention 用 Triton 分块算同一条式子，MLA 再把内容压进一条低秩向量。

MHA 和 MLA 的宽度用 DeepSeek-V2：D=5120，H=128，d=128。GQA 在同一宽度上把 KV 头收成 8。

## MHA

每个头一套投影，RoPE 打在整头上。注意力就是上面那条式子，matmul 加 softmax，不经过 Flash Attention。

```text
Q = RoPE(Wq h)    # [B, S, H, d]
K = RoPE(Wk h)
V = Wv h
```

```python
import torch
import torch.nn as nn


def rope(x):
    # x: [B, S, H, D]，D 为偶数。只旋转最后一维。
    s, d = x.shape[-3], x.shape[-1]
    half = d // 2
    pos = torch.arange(s, device=x.device)
    inv_freq = 1.0 / (10000 ** (torch.arange(half, device=x.device) / half))
    angles = pos[:, None] * inv_freq[None, :]
    cos = angles.cos().view(1, s, 1, half)
    sin = angles.sin().view(1, s, 1, half)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


def attention(q, k, v):
    # q, k, v: [B, H, S, D]。分数矩阵 [B, H, S, S] 直接落地。
    s = q.shape[-2]
    scores = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    pos = torch.arange(s, device=q.device)
    scores = scores.masked_fill(pos[None, :] > pos[:, None], float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, head_dim):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, head_dim
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(self, x):
        b, s, _ = x.shape
        h, d = self.n_heads, self.head_dim
        q = rope(self.wq(x).view(b, s, h, d))
        k = rope(self.wk(x).view(b, s, h, d))
        v = self.wv(x).view(b, s, h, d)
        # 解码缓存: k, v → [B, S, H, d] 各一份
        y = attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.wo(y.transpose(1, 2).reshape(b, s, h * d))
```

`Wq/Wk/Wv` 各是 D×(H×d)，`Wo` 是 (H×d)×D。这一层参数量 4×5120×128×128 = **3.36×10^8**。

解码时历史 token 的 K、V 都要留着，每个 token 是 2×H×d = 2×128×128 = **32768** 个数。S 一长，缓存按头数线性涨。

## GQA

Q 仍然是 H 头。K、V 只投影 `n_kv_heads` 头，`H` 必须整除 `n_kv_heads`。一组里有 `n_rep = H / n_kv_heads` 个 query 头，算注意力前把每份 K、V 在头维重复 `n_rep` 次，后面的 matmul 和 MHA 相同。`n_kv_heads = H` 时就是上一节；`n_kv_heads = 1` 时是 MQA，所有 query 头共用一份 K、V。

```text
Q = RoPE(Wq h)                 # [B, S, H, d]
K = RoPE(Wk h)                 # [B, S, n_kv_heads, d]
V = Wv h
K = repeat(K, n_rep)           # 重复后才是 [B, S, H, d]
```

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, head_dim):
        super().__init__()
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
        self.n_rep = n_heads // n_kv_heads
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(self, x):
        b, s, _ = x.shape
        h, kv, d = self.n_heads, self.n_kv_heads, self.head_dim
        q = rope(self.wq(x).view(b, s, h, d))
        k = rope(self.wk(x).view(b, s, kv, d))
        v = self.wv(x).view(b, s, kv, d)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        # 缓存留在 repeat 之前: k, v → [B, n_kv_heads, S, d]
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        y = attention(q, k, v)
        return self.wo(y.transpose(1, 2).reshape(b, s, h * d))
```

`n_kv_heads = 8` 时，16 个 query 头共用一份 K、V。`Wk/Wv` 从 D×(128×128) 收成 D×(8×128)。这一层参数量 **1.78×10^8**。每个 token 的缓存是 2×8×128 = **2048** 个数。

## Flash Attention

算的是和 `attention` 同一条 `softmax(QK^T)V`。S×S 的 scores 不写出内存，Q、K、V 分块走。每一行只留三个运行量：当前最大值 m、指数和 ℓ、还没除进去的输出 O。换下一块 KV 时，用新旧最大值把旧的 ℓ 和 O 缩一下再累加：

```text
m_new = max(m, rowmax(S_blk))
alpha = exp(m - m_new)
ℓ     = alpha * ℓ + rowsum(exp(S_blk - m_new))
O     = alpha * O + exp(S_blk - m_new) @ V_blk
# 块扫完: O = O / ℓ
```

下面是 Triton 里的因果前向。一个 program 管一个 query 块（`BLOCK_M` 行）和一对 `(batch, head)`。Q、K、V 要是 fp16 或 bf16，`HEAD_DIM` 取 16 的倍数，常用 64 或 128。GQA 可以像上一节那样先 repeat 再启动；要省掉这块重复流量，KV 头下标用 `off_h // n_rep`。

```python
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(Q, K, V, Out, sm_scale,
              stride_qb, stride_qh, stride_qm, stride_qd,
              stride_kb, stride_kh, stride_kn, stride_kd,
              stride_vb, stride_vh, stride_vn, stride_vd,
              stride_ob, stride_oh, stride_om, stride_od,
              H, N_CTX,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    q_off = off_z.to(tl.int64) * stride_qb + off_h.to(tl.int64) * stride_qh
    k_off = off_z.to(tl.int64) * stride_kb + off_h.to(tl.int64) * stride_kh
    v_off = off_z.to(tl.int64) * stride_vb + off_h.to(tl.int64) * stride_vh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                mask=offs_m[:, None] < N_CTX, other=0.0)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_k = start_n + offs_n
        k = tl.load(K + k_off + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                    mask=offs_k[:, None] < N_CTX, other=0.0)
        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale
        qk = tl.where(offs_k[None, :] < N_CTX, qk, float("-inf"))
        qk = tl.where(offs_m[:, None] >= offs_k[None, :], qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + v_off + offs_k[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                    mask=offs_k[:, None] < N_CTX, other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    tl.store(Out + q_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
             acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def flash_attention(q, k, v):
    # q, k, v: [B, H, S, D]，fp16 或 bf16
    b, h, s, d = q.shape
    out = torch.empty_like(q)
    grid = (triton.cdiv(s, 64), b * h)
    _attn_fwd[grid](
        q, k, v, out, d ** -0.5,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        h, s,
        BLOCK_M=64, BLOCK_N=64, HEAD_DIM=d,
    )
    return out
```

`alpha` 把上一块以旧最大值算出的 exp 对齐到新最大值。扫完除以 ℓ，就是这一行 softmax 的分母。内层循环只走到当前 query 块的右边界 `(start_m + 1) * BLOCK_M`，后面的 KV 块在因果掩码下全用不到。

省的是训练时那张 **S×S** 分数矩阵。投影还是 MHA 或 GQA，缓存大小不变。

## MLA

K、V 的内容不按头各存一份。h 先压成低秩 `c_KV`（秩 r=512），各头的内容 K 和 V 都从它乘出来。位置不进 `c_KV`：另外做一条所有头共享的短 key `k^R`（宽 d_rope=64），RoPE 只打在这里。Q 同样先压到秩 1536，再拆成内容 q^C 和 RoPE 部分 q^R。

```text
c_KV     = W_DKV h                 # [B, S, 512]
K_nope   = W_UK c_KV               # [B, S, H, 128]
V        = W_UV c_KV               # [B, S, H, 128]
k^R      = RoPE(W_KR h)            # [B, S, 64]，所有头共用
Q        = [q^C ; q^R]             # 每头 128+64
K        = [K_nope ; k^R]
# scores = Q K^T / sqrt(128+64)
```

```python
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, dim, n_heads, nope_dim, rope_dim, v_dim, kv_rank, q_rank):
        super().__init__()
        self.n_heads = n_heads
        self.nope_dim, self.rope_dim, self.v_dim = nope_dim, rope_dim, v_dim
        self.w_dkv = nn.Linear(dim, kv_rank, bias=False)
        self.w_uk = nn.Linear(kv_rank, n_heads * nope_dim, bias=False)
        self.w_uv = nn.Linear(kv_rank, n_heads * v_dim, bias=False)
        self.w_kr = nn.Linear(dim, rope_dim, bias=False)
        self.w_dq = nn.Linear(dim, q_rank, bias=False)
        self.w_uq = nn.Linear(q_rank, n_heads * nope_dim, bias=False)
        self.w_qr = nn.Linear(q_rank, n_heads * rope_dim, bias=False)
        self.w_o = nn.Linear(n_heads * v_dim, dim, bias=False)

    def forward(self, x):
        b, s, _ = x.shape
        h = self.n_heads
        c_kv = self.w_dkv(x)  # [B, S, kv_rank]，解码时缓存这个
        k_nope = self.w_uk(c_kv).view(b, s, h, self.nope_dim)
        v = self.w_uv(c_kv).view(b, s, h, self.v_dim)
        c_q = self.w_dq(x)
        q_nope = self.w_uq(c_q).view(b, s, h, self.nope_dim)
        q_pe = rope(self.w_qr(c_q).view(b, s, h, self.rope_dim))
        k_pe = rope(self.w_kr(x).unsqueeze(2)).expand(b, s, h, self.rope_dim)
        q = torch.cat((q_nope, q_pe), dim=-1)
        k = torch.cat((k_nope, k_pe), dim=-1)  # 内容来自低秩，位置来自共享 k_pe
        # 解码缓存: c_kv [B, S, kv_rank]，共享 k^R [B, S, rope_dim]
        # 这里用和 MHA 相同的 matmul。QK 宽 192、V 宽 128，分块时换成上一节的 kernel。
        y = attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.w_o(y.transpose(1, 2).reshape(b, s, h * self.v_dim))
```

`rope` 和 `attention` 是 MHA 那一节的函数。点积宽度是 **192**，缩放换成 `1/sqrt(192)`。V 仍是 128，所以一头的输出宽度跟 K 的宽度不必相同。上一节的 kernel 把 QK 宽度和 V 宽度写成同一个 `HEAD_DIM`；MLA 要分成 192 和 128 两套。

解码缓存是 `c_KV` 和 `k^R`：512+64 = **576** 个数。同一宽度下 MHA 是 32768，8 个 KV 头的 GQA 是 2048。这一层投影参数是 **1.49×10^8**（含 Q 的 1536 低秩、KV 的 512 低秩，以及把低秩乘回头宽的那几块）。

RoPE 放在 `c_KV` 里面会和后面的线性层缠在一起：先乘矩阵再旋转，和先旋转再乘矩阵，结果不一样。位置留在 `k^R` 上，`c_KV` 才能按 token 缓存、用到时再投影。

解码时内容分不必先乘成 H 份 K。`(W_UK^T q^C)` 直接和缓存里的 `c_KV` 做点积，`W_UV` 留到加权之后。`k^R` 仍和 `q^R` 单独点积，加进同一个 score。

---

softmax 那几节的缓存跟着 S 涨。Linear attention 把历史收成一块固定状态 `S`，每头一块 `d×d`，每个 token 的更新和读取都是 O(d²)。同一宽度 H=128、d=128 时，一层状态是 128×128×128 = **2.10×10⁶** 个数，和 S 无关。MHA 在 S=4096 时缓存是 32768×4096 = **1.34×10⁸**。下面几条都是在改 `S` 怎么写。

## Linear Attention

`softmax(QK^T)V` 不能把乘法交换顺序。去掉 softmax，用特征映射 φ 把相似度拆成 φ(q)·φ(k)，就能先把 K 和 V 乘成状态，Q 只负责读：

```text
S_t = S_{t-1} + φ(k_t)^T v_t
z_t = z_{t-1} + φ(k_t)
y_t = φ(q_t) S_t / (φ(q_t)·z_t)
```

φ 用 `elu(x)+1`，保证是正的。`S` 是 `[Dk, Dv]`，`z` 是归一化用的分母。因果掩码变成按时间累加，解码只留 `S` 和 `z`。

```python
def linear_attention(q, k, v):
    # q, k: [B, H, S, Dk]，v: [B, H, S, Dv]
    q, k = torch.nn.functional.elu(q) + 1, torch.nn.functional.elu(k) + 1
    b, h, s, dk = q.shape
    state = q.new_zeros(b, h, dk, v.shape[-1])
    z = q.new_zeros(b, h, dk)
    out = []
    for t in range(s):
        kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
        state = state + kt.unsqueeze(-1) * vt.unsqueeze(-2)
        z = z + kt
        num = (qt.unsqueeze(-2) @ state).squeeze(-2)
        den = (qt * z).sum(-1, keepdim=True) + 1e-6
        out.append(num / den)
    return torch.stack(out, dim=2)
```

状态只加不减，旧的外积一直留在 `S` 里。后面几条都是给这个加法加衰减或覆盖。

## Lightning Attention-2

衰减是一个固定的 λ，0<λ<1，每个头可以各有一个。逐 token 就是 `S_t = λ S_{t-1} + k_t^T v_t`，`y_t = q_t S_t`。TransNormerLLM 和 MiniMax 用的是它的分块算法：块内用带衰减的左乘，块间用右乘去读上一块留下的 `S`。

```text
M_ij = λ^{i-j}    i >= j，否则 0
O_intra = (Q K^T ⊙ M) V
O_inter = (Q ⊙ λ^{pos+1}) S_prev
S       = λ^n S_prev + (K ⊙ λ^{n-1-pos})^T V
```

`pos` 是块内下标，`n` 是块长。`λ^{pos+1}` 是因为 `S_prev` 已经是上一 token 写完的状态，当前 token 读之前还要再衰减一次。

```python
def lightning_attention(q, k, v, lam, block=64):
    # q, k: [B, H, S, Dk]，v: [B, H, S, Dv]，lam 为 Python 浮点
    b, h, s, dk = q.shape
    state = q.new_zeros(b, h, dk, v.shape[-1])
    outs = []
    for i in range(0, s, block):
        qb, kb, vb = q[:, :, i:i + block], k[:, :, i:i + block], v[:, :, i:i + block]
        n = qb.shape[2]
        pos = torch.arange(n, device=q.device)
        dist = (pos[:, None] - pos[None, :]).clamp(min=0)
        mask = lam ** dist
        mask = mask.masked_fill(pos[:, None] < pos[None, :], 0)
        intra = torch.matmul(torch.matmul(qb, kb.transpose(-2, -1)) * mask, vb)
        inter = torch.matmul(qb * (lam ** (pos + 1)).view(1, 1, n, 1), state)
        outs.append(intra + inter)
        w = lam ** (n - 1 - pos)
        state = (lam ** n) * state + torch.matmul(
            (kb * w.view(1, 1, n, 1)).transpose(-2, -1), vb,
        )
    return torch.cat(outs, dim=2)
```

λ 不看输入。所有通道、所有 token 用同一个衰减。

## GLA

Gated Linear Attention 把 λ 换成每个 token、每个通道各一个 α，由当前输入算出来，落在 (0, 1)。衰减乘在 key 那一维上：

```text
α_t = σ(W_α x_t)          # [Dk]
S_t = Diag(α_t) S_{t-1} + k_t^T v_t
y_t = q_t S_t
```

```python
def gated_linear_attention(q, k, v, alpha):
    # alpha: [B, H, S, Dk]，每个通道一个遗忘门
    b, h, s, dk = q.shape
    state = q.new_zeros(b, h, dk, v.shape[-1])
    out = []
    for t in range(s):
        state = state * alpha[:, :, t].unsqueeze(-1)
        state = state + k[:, :, t].unsqueeze(-1) * v[:, :, t].unsqueeze(-2)
        out.append((q[:, :, t].unsqueeze(-2) @ state).squeeze(-2))
    return torch.stack(out, dim=2)
```

和 Lightning 的差别只在 α 是向量，而且每个位置重新算。训练时这一递推写成按块的并行扫描，数学相同。

## KDA

Kimi Linear 的 Kimi Delta Attention。在 GLA 的逐通道衰减上再加 DeltaNet 的覆盖：先把旧状态按 α 衰减，再沿着当前 key 的方向，用 `v_t - k_t S` 把这条关联改写成新的 v。k 先做 L2 归一化，β 是 (0, 1) 的标量写入强度。

```text
k_t = k_t / ||k_t||
S'  = Diag(α_t) S_{t-1}
S_t = S' + β_t k_t^T (v_t - k_t S')
y_t = q_t S_t
```

β=1 且 k 是单位向量时，`k_t S_t = v_t`，这个方向上的旧值被盖掉。α 退化为标量时就是 Gated DeltaNet。

```python
def kimi_delta_attention(q, k, v, alpha, beta):
    # alpha: [B, H, S, Dk]，beta: [B, H, S]
    k = torch.nn.functional.normalize(k, dim=-1)
    b, h, s, dk = q.shape
    state = q.new_zeros(b, h, dk, v.shape[-1])
    out = []
    for t in range(s):
        decayed = state * alpha[:, :, t].unsqueeze(-1)
        kt, vt = k[:, :, t], v[:, :, t]
        pred = torch.einsum("bhd,bhdv->bhv", kt, decayed)
        state = decayed + torch.einsum("bh,bhd,bhv->bhdv", beta[:, :, t], kt, vt - pred)
        out.append(torch.einsum("bhd,bhdv->bhv", q[:, :, t], state))
    return torch.stack(out, dim=2)
```

Kimi Linear 是混合堆叠：一部分层用 KDA，一部分层仍是 softmax attention。状态形状和上面几条一样，仍是每头一块 `Dk×Dv`。
