import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import *


def ifnone(a, b):
    # From fastai.fastcore
    "`b` if `a` is None else `a`"
    return b if a is None else a


class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int, res_attention: bool = False):
        super().__init__()
        self.d_k, self.res_attention = d_k, res_attention

    def forward(self, q, k, v, prev=None, attn_mask=None):

        # MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        scores = torch.matmul(q, k)                                    # scores : [bs x n_heads x q_len x q_len]

        # Scale
        scores = scores / (self.d_k ** 0.5)

        # Attention mask (optional)
        if attn_mask is not None:                                     # mask with shape [q_len x q_len]
            if attn_mask.dtype == torch.bool:
                scores.masked_fill_(attn_mask, float('-inf'))
            else:
                scores += attn_mask

        # SoftMax
        if prev is not None: scores = scores + prev

        attn = F.softmax(scores, dim=-1)                               # attn   : [bs x n_heads x q_len x q_len]

        # MatMul (attn, v)
        context = torch.matmul(attn, v)                                # context: [bs x n_heads x q_len x d_v]

        if self.res_attention:
            return context, attn, scores
        else:
            return context, attn


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int, res_attention: bool = False):
        """Input shape:  Q, K, V:[batch_size (bs) x q_len x d_model], mask:[q_len x q_len]"""
        super().__init__()
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)

        self.W_O = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.res_attention = res_attention

        # Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            self.sdp_attn = _ScaledDotProductAttention(self.d_k, self.res_attention)
        else:
            self.sdp_attn = _ScaledDotProductAttention(self.d_k)

    def forward(self, Q, K, V, prev=None, attn_mask=None):

        bs = Q.size(0)

        # Linear (+ split in multiple heads)
        # q_s = self.W_Q(Q)
        # print("1", q_s)
        # k_s = self.W_K(K)
        # print("2", k_s)
        # v_s = self.W_V(V)
        # print("3", v_s)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)       # q_s    : [bs x n_heads x q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            context, attn, scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, attn_mask=attn_mask)
        else:
            context, attn = self.sdp_attn(q_s, k_s, v_s, attn_mask=attn_mask)
        # context: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len]

        # Concat
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # context: [bs x q_len x n_heads * d_v]

        # Linear
        output = self.W_O(context)
        # print("output", output.shape)

        if self.res_attention:
            return output, attn, scores, context
        else:
            return output, attn                                                          # output: [bs x q_len x d_model]


class _TabEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                 res_dropout=0.1, activation="gelu", res_attention=False):

        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = ifnone(d_k, d_model // n_heads)
        d_v = ifnone(d_v, d_model // n_heads)
        d_ff = ifnone(d_ff, d_model * 4)

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(res_dropout)
        self.layernorm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), self._get_activation_fn(activation), nn.Linear(d_ff, d_model))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(res_dropout)
        self.layernorm_ffn = nn.LayerNorm(d_model)

    def forward(self, src, prev=None, attn_mask=None):

        # Multi-Head attention sublayer
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores, context = self.self_attn(src, src, src, prev, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, attn_mask=attn_mask)
        self.attn = attn
        ## Add & Norm
        # src = src.view(-1, 10, 20)
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        src = self.layernorm_attn(src) # Norm: layernorm

        # Feed-forward sublayer
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        src = self.layernorm_ffn(src) # Norm: layernorm

        if self.res_attention:
            return src, context
        else:
            return src

    def _get_activation_fn(self, activation):
        if callable(activation):
            return activation()
        elif activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "gelu":
            return nn.GELU()
        raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')


class _TabEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, res_dropout=0.1, activation='relu', res_attention=False, n_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([_TabEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, res_dropout=res_dropout,
                                                            activation=activation, res_attention=res_attention) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src, attn_mask=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, context = mod(output, prev=scores, attn_mask=attn_mask)
            return output, context
        else:
            for mod in self.layers:
                output = mod(output, attn_mask=attn_mask)
            return output


class Net(nn.Module):
    def __init__(self, c_in, c_out, d_model=96, n_layers=1, n_heads=1, d_k=None, d_v=None, d_ff=None, res_attention=True,
                 attention_act='gelu', res_dropout=0.1):
        super(Net, self).__init__()

        self.inlinear = nn.Linear(c_in, d_model)
        self.relu = nn.GELU()
        self.trans_encoder = _TabEncoder(16, d_model, n_heads=n_heads,  d_k=d_k, d_v=d_v, d_ff=d_ff,
                                         res_dropout=res_dropout, activation=attention_act,
                                         res_attention=res_attention, n_layers=n_layers)

        self.permute = Permute(0, 2, 1)
        self.transpose = Transpose(0, 1)
        self.max = Max(1)
        self.outlinear = nn.Linear(d_model, c_out)
        print("Number Parameters: ", self.get_n_params())

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def forward(self, x):
        # x = self.permute(x)
        # print(x.shape)
        # x = x.view(-1, 10, 20)
        x = self.inlinear(x)
        x = self.relu(x)
        en_x, context = self.trans_encoder(x)
        # print("3", context.shape)
        # x = self.transpose(x)
        x = self.max(en_x)
        x = self.relu(x)
        x = self.outlinear(x)
        return x, context

