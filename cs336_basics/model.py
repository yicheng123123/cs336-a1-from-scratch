import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(d_out, d_in)
        )

        # CS336 initialization
        std = (2 / (d_in + d_out)) ** 0.5

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )
    def forward (self,x):
        return x @self.weight.T
    
class Embedding (nn.Module):
    def __init__(self,num_embeddings, embedding_dim):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
            )
        )

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )
    def forward(self,token_ids):
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self,d_model,eps=1e-5):
        super().__init__()
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        original_dtype = x.dtype

        x_float = x.to(torch.float32)

        rms_inv = torch.rsqrt(
            torch.mean(
                x_float ** 2,
                dim=-1,
                keepdim=True
            )
            + self.eps
        )

        output = (
            x_float
            * rms_inv
            * self.weight
        )
        return output.to(original_dtype)
import torch.nn.functional as F
class SwiGLU(nn.Module):
    def __init__(self,d_model,d_ff):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
    def forward (self,x):
        gate=F.silu(self.w1(x))
        value=self.w3(x)
        return self.w2(gate*value)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,theta,d_k,max_seq_len,device=None):
        super().__init__(
        )
        assert d_k%2==0
        i=torch.arange(d_k//2,device=device,dtype=torch.float32)
        freqs = theta ** (
            -2 * i / d_k
        )
        positions=torch.arange(max_seq_len,device=device,dtype=torch.float32)
        angles = (
            positions[:, None]
            * freqs[None, :]
        )
        self.register_buffer(
            "cos_cache",
            torch.cos(angles),
            persistent=False,
        )

        self.register_buffer(
            "sin_cache",
            torch.sin(angles),
            persistent=False,
        )
    def forward(self,
        x,
        token_positions,):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        rot_even = (
            x_even * cos
            - x_odd * sin
        )

        rot_odd = (
            x_even * sin
            + x_odd * cos
        )

        rotated = torch.stack(
            [rot_even, rot_odd],
            dim=-1,
        )

        return rotated.flatten(-2)
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]

    scores = Q @ K.transpose(-2, -1)
    scores = scores / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(
            ~mask,
            float("-inf")
        )

    attention_weights = torch.softmax(
    scores,
    dim=-1
)
    output=attention_weights @ V

    return output

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads,theta=None,
        max_seq_len=None,):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj=Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.rope = None

        if theta is not None:
            assert max_seq_len is not None

            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.head_dim,
                max_seq_len=max_seq_len,
            )
    def forward(self,x, token_positions=None):
        B,T,_=x.shape
        Q=self.q_proj(x)
        K=self.k_proj(x)
        V=self.v_proj(x)
        Q=Q.view(B,T,self.num_heads,self.head_dim,
        ).transpose(1, 2)
        K = K.view(
            B,
            T,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        V = V.view(
            B,
            T,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(
                    T,
                    device=x.device,
                )

            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        mask=torch.tril(torch.ones(T,T,dtype=torch.bool,device=x.device,))
        out = scaled_dot_product_attention(
            Q,
            K,
            V,
            mask,
        )
        out=out.transpose(1,2)
        out = out.reshape(B,T,self.d_model,)

        return self.output_proj(out)
    
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        theta,
        max_seq_len,
    ):
        super().__init__()

        self.ln1 = RMSNorm(d_model)

        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
        )

        self.ln2 = RMSNorm(d_model)

        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
        )
    def forward(self, x, token_positions=None):

        x = x + self.attn(
            self.ln1(x),
            token_positions=token_positions,
        )

        x = x + self.ffn(
            self.ln2(x)
        )

        return x
class TransformerLM(nn.Module):
    def __init__(self,vocab_size,context_length,d_model,num_layers,num_heads,d_ff,rope_theta,):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length

        self.token_embeddings = Embedding(
            vocab_size,
            d_model,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=rope_theta,
                max_seq_len=context_length,
            )
            for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model)

        self.lm_head = Linear(
            d_model,
            vocab_size,
        )
    def forward(self, token_ids):
        x = self.token_embeddings(token_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)

        logits = self.lm_head(x)

        return logits