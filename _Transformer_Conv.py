import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
import torchvision.utils as tvutils
import matplotlib.pyplot as plt
import seaborn as sns
from .layers import *

length = 20
dimension = 10
writer = SummaryWriter('./results/visualization')



def ifnone(a, b):
    # From fastai.fastcore
    "`b` if `a` is None else `a`"
    return b if a is None else a


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int, res_attention: bool = False):
        super().__init__()
        self.d_k, self.res_attention = d_k, res_attention

    def forward(self, q, k, v, prev=None, attn_mask=None):

        # MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        # mask = 1 - torch.linalg()
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
        if prev is not None:
            scores = scores + prev

        attn = F.softmax(scores, dim=-1)                               # attn   : [bs x n_heads x q_len x q_len]

        # MatMul (attn, v)
        context = torch.matmul(attn, v)                                # context: [bs x n_heads x q_len x d_v]

        if self.res_attention:
            return context, attn, scores
        else:
            return context, attn


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int, res_attention: bool = False, dropout=0.1):
        """Input shape:  Q, K, V:[batch_size (bs) x q_len x d_model], mask:[q_len x q_len]"""
        super().__init__()

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        # self.W_Q1 = nn.Linear(d_model, d_k * n_heads, bias=False)
        # self.W_K1 = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.v_fc = nn.Linear(dimension, d_model)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # self.conv1 = nn.Conv1d(dimension, 16, kernel_size=3)
        # self.conv1_1 = nn.Conv1d(16, 32, kernel_size=3)
        # self.conv1_2 = nn.Conv1d(32, 48, kernel_size=3)
        # self.conv1_3 = nn.Conv1d(48, 64, kernel_size=3)
        # self.conv1_4 = nn.Conv1d(64, 80, kernel_size=3)
        # self.conv1_5 = nn.Conv1d(80, 90, kernel_size=3)
        # self.conv1_6 = nn.Conv1d(90, d_k * n_heads, kernel_size=3)
        # self.conv2 = nn.Conv1d(dimension, 16, kernel_size=3)
        # self.conv2_1 = nn.Conv1d(16, 32, kernel_size=3)
        # self.conv2_2 = nn.Conv1d(32, 48, kernel_size=3)
        # self.conv2_3 = nn.Conv1d(48, 64, kernel_size=3)
        # self.conv2_4 = nn.Conv1d(64, 80, kernel_size=3)
        # self.conv2_5 = nn.Conv1d(80, 90, kernel_size=3)
        # self.conv2_6 = nn.Conv1d(90, d_k * n_heads, kernel_size=3)

        # self.conv1 = nn.Conv1d(dimension, 32, kernel_size=3)
        # self.conv1_1 = nn.Conv1d(32, 64, kernel_size=3)
        # self.conv1_2 = nn.Conv1d(64, d_k * n_heads, kernel_size=3)
        #
        # self.conv2 = nn.Conv1d(dimension, 32, kernel_size=3)
        # self.conv2_1 = nn.Conv1d(32, 64, kernel_size=3)
        # self.conv2_2 = nn.Conv1d(64, d_k * n_heads, kernel_size=3)
        # self.W_Q = nn.Sequential(self.conv1, self.conv1_1, self.conv1_2)
        # self.W_K = nn.Sequential(self.conv2, self.conv2_1, self.conv2_2)

        self.conv1 = nn.Conv1d(dimension, dimension, kernel_size=3, padding=2, groups=dimension)
        self.conv1_5 = nn.Conv1d(dimension, d_k * n_heads, kernel_size=1)
        # self.conv1_5 = nn.Conv1d(10, d_k * n_heads, kernel_size=3, padding=2)

        self.conv2 = nn.Conv1d(dimension, dimension, kernel_size=3, padding=2, groups=dimension)
        self.conv2_5 = nn.Conv1d(dimension, d_k * n_heads, kernel_size=1)
        # self.conv2_5 = nn.Conv1d(10, d_k * n_heads, kernel_size=3, padding=2)

        self.W_Q = nn.Sequential(self.conv1, self.conv1_5)
        self.W_K = nn.Sequential(self.conv2, self.conv2_5)

        # self.conv1 = weight_norm(nn.Conv1d(dimension, 32, kernel_size=3, padding=2, dilation=1))
        # self.conv1_1 = weight_norm(nn.Conv1d(32, 64, kernel_size=3, padding=4, dilation=2))
        # self.conv1_2 = weight_norm(nn.Conv1d(64, d_k * n_heads, kernel_size=3, padding=8, dilation=4))
        # self.conv2 = weight_norm(nn.Conv1d(dimension, 32, kernel_size=3, padding=2, dilation=1))
        # self.conv2_1 = weight_norm(nn.Conv1d(32, 64, kernel_size=3, padding=4, dilation=2))
        # self.conv2_2 = weight_norm(nn.Conv1d(64, d_k * n_heads, kernel_size=3, padding=8, dilation=4))
        # self.chomp1 = Chomp1d(2)
        # self.chomp2 = Chomp1d(4)
        #
        # self.W_Q = nn.Sequential(self.conv1, self.chomp1, self.relu, self.dropout,
        #                          self.conv1_1, self.chomp2, self.relu, self.dropout,
        #                          self.conv1_2, self.chomp2, self.relu, self.dropout)
        # self.W_K = nn.Sequential(self.conv2, self.chomp1, self.relu, self.dropout,
        #                          self.conv2_1, self.chomp2, self.relu, self.dropout,
        #                          self.conv2_2, self.chomp2, self.relu, self.dropout)

        # -----------------------------------------------------------------------------
        # self.conv1 = weight_norm(nn.Conv1d(dimension, dimension, kernel_size=3, padding=2, dilation=1, groups=dimension))
        # self.conv1_o = weight_norm(nn.Conv1d(dimension, 32, kernel_size=1))
        # self.conv1_1 = weight_norm(nn.Conv1d(32, 32, kernel_size=3, padding=4, dilation=2, groups=32))
        # self.conv1_1o = weight_norm(nn.Conv1d(32, 64, kernel_size=1))
        # self.conv1_2 = weight_norm(nn.Conv1d(64, 64, kernel_size=3, padding=8, dilation=4, groups=64))
        # self.conv1_2o = weight_norm(nn.Conv1d(64, d_k * n_heads, kernel_size=1))
        #
        # self.conv2 = weight_norm(nn.Conv1d(dimension, dimension, kernel_size=3, padding=2, dilation=1, groups=dimension))
        # self.conv2_o = weight_norm(nn.Conv1d(dimension, 32, kernel_size=1))
        # self.conv2_1 = weight_norm(nn.Conv1d(32, 32, kernel_size=3, padding=4, dilation=2, groups=32))
        # self.conv2_1o = weight_norm(nn.Conv1d(32, 64, kernel_size=1))
        # self.conv2_2 = weight_norm(nn.Conv1d(64, 64, kernel_size=3, padding=8, dilation=4, groups=64))
        # self.conv2_2o = weight_norm(nn.Conv1d(64, d_k * n_heads, kernel_size=1))
        # self.chomp1 = Chomp1d(2)
        # self.chomp2 = Chomp1d(4)
        #
        # self.W_Q = nn.Sequential(self.conv1, self.conv1_o, self.relu, self.dropout,
        #                          self.conv1_1, self.conv1_1o, self.relu, self.dropout,
        #                          self.conv1_2, self.conv1_2o, self.relu, self.dropout)
        # self.W_K = nn.Sequential(self.conv2, self.conv2_o, self.relu, self.dropout,
        #                          self.conv2_1, self.conv2_1o, self.relu, self.dropout,
        #                          self.conv2_2, self.conv2_2o, self.relu, self.dropout)

        # self.W_Q = nn.Conv1d(1, d_k * n_heads, kernel_size=(5, 5), padding=(4, 4), dilation=(1, 1))
        # self.W_K = nn.Conv1d(1, d_k * n_heads, kernel_size=(5, 5), padding=(4, 4), dilation=(1, 1))
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)

        self.W_O = nn.Linear(n_heads * d_v, d_model, bias=False)
        # self.relu = nn.ReLU()
        # self.fc = nn.Linear(d_model, d_model)

        self.res_attention = res_attention

        # self.init_weights()

        # Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            self.sdp_attn = _ScaledDotProductAttention(self.d_k, self.res_attention)
        else:
            self.sdp_attn = _ScaledDotProductAttention(self.d_k)

    def init_weights(self):
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         torch.nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, nn.Linear):
        #         torch.nn.init.kaiming_normal_(m.weight)

        self.conv1.weight.data.normal_(0, 0.01)
        self.conv1_1.weight.data.normal_(0, 0.01)
        self.conv1_2.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        self.conv2_1.weight.data.normal_(0, 0.01)
        self.conv2_2.weight.data.normal_(0, 0.01)

    def forward(self, Q, K, V, prev=None, attn_mask=None):

        bs = Q.size(0)

        # Linear (+ split in multiple heads)
        # x = self.conv1(Q)
        # q_s = self.W_Q(Q)
        # print("qsshape", q_s.shape)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)     # q_s    : [bs x n_heads x q_len x d_k]
        # print("qsnewshape", q_s.shape)
        # k_s = self.W_K(K)
        # print(k_s.shape)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        V = self.v_fc(V.view(-1, length, dimension))
        # v_s = self.W_V(V)
        # print(v_s.shape)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).permute(0, 2, 3, 1)     # v_s    : [bs x n_heads x q_len x d_v]
        # print("vsshape", v_s.shape)
        # Scaled Dot-Product Attention (multiple heads)

        # q_s1 = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)  # q_s    : [bs x n_heads x q_len x d_k]
        # k_s1 = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        # v_s1 = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)

        if self.res_attention:
            context, attn, scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, attn_mask=attn_mask)
            # context1, _, _ = self.sdp_attn(q_s1, k_s1, v_s1, prev=prev, attn_mask=attn_mask)
        else:
            context, attn = self.sdp_attn(q_s, k_s, v_s, attn_mask=attn_mask)
            # context1, _ = self.sdp_attn(q_s1, k_s1, v_s1, attn_mask=attn_mask)
        # context: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len]

        # Concat
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # context: [bs x q_len x n_heads * d_v]
        # context1 = context.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        # contextc = context[0]
        # torch.save(contextc, "contextc.pt")
        # contextl = context1[0]
        # torch.save(contextl, "contextl.pt")

        # attn = attn[0][0]
        # torch.save(attn, "similarity.pt")

        # print(attn.shape)
        # img_grid = tvutils.make_grid(context1, normalize=True, scale_each=True, nrow=2)
        # writer.add_image('attention_img', img_grid, global_step=6)
        # print(context.shape)
        # Linear
        output = self.W_O(context)
        # print("output", output.shape)

        if self.res_attention:
            return output, attn, scores
        else:
            return output, attn                                                           # output: [bs x q_len x d_model]


class _TabEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                 res_dropout=0.1, activation="gelu", res_attention=False):

        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = ifnone(d_k, d_model // n_heads)
        d_v = ifnone(d_v, d_model // n_heads)
        d_ff = ifnone(d_ff, d_model * 4)

        self.in_fc = nn.Linear(dimension, d_model)
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
            src2, attn, scores = self.self_attn(src, src, src, prev, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, attn_mask=attn_mask)
        self.attn = attn
        ## Add & Norm
        # print(src.shape, src2.shape)
        src = src.view(-1, length, dimension)
        src = self.in_fc(src)
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        # print("srcshape", src.shape)
        src = self.layernorm_attn(src) # Norm: layernorm

        # Feed-forward sublayer
        ## Position-wise Feed-Forward
        # src2 = src.view(-1, dimension, length)
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
        src = self.layernorm_ffn(src)  # Norm: layernorm

        if self.res_attention:
            return src, scores
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
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, res_dropout=0.1, activation='gelu', res_attention=False, n_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([_TabEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, res_dropout=res_dropout,
                                                            activation=activation, res_attention=res_attention) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src, attn_mask=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(output, prev=scores, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers:
                output = mod(output, attn_mask=attn_mask)
            return output


class Net(nn.Module):
    def __init__(self, c_in, c_out, d_model=96, n_layers=1, n_heads=4, d_k=None, d_v=None, d_ff=None, res_attention=True,
                 attention_act='relu', res_dropout=0.1):
        super(Net, self).__init__()

        self.inlinear = nn.Linear(c_in, d_model)
        self.relu = nn.ReLU()
        self.trans_encoder = _TabEncoder(1, d_model, n_heads=n_heads,  d_k=d_k, d_v=d_v, d_ff=d_ff,
                                         res_dropout=res_dropout, activation=attention_act,
                                         res_attention=res_attention, n_layers=n_layers)

        self.transpose = Transpose(0, 1)
        self.max = Max(1)
        self.outlinear = nn.Linear(d_model, c_out)

        self.initialize_weights()
        print(self)
        print("Number Parameters: ", self.get_n_params())

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def initialize_weights(self):
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight, gain=1)
        #         m.bias.data.zero_()

        initrange = 0.1
        self.outlinear.bias.data.zero_()
        self.outlinear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # x = self.inlinear(x)
        # x = self.relu(x)
        x = x.view(-1, dimension, length)
        # en_x, context = self.trans_encoder(x)
        en_x = self.trans_encoder(x)
        # print("xshape", context.shape)
        # x = self.transpose(x)
        x = self.max(en_x)
        x = self.relu(x)
        x = self.outlinear(x)
        return x

