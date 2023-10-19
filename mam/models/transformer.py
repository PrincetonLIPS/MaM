import math
import torch
from torch import nn, Tensor
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F

from models.nn import MLP


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(d_model//2) * (-math.log(10000.0) / (d_model//2 - 1)) )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, : d_model//2] = torch.sin(position * div_term)
        pe[0, :, d_model//2 : ] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe
        return x

class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, nembed, nhead, dropout):
        super().__init__()
        # key, query, value projections for all heads
        self.key = nn.Linear(nembed, nembed, bias=False)
        self.query = nn.Linear(nembed, nembed, bias=False)
        self.value = nn.Linear(nembed, nembed, bias=False)
        # regularization
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # output projection
        self.proj = nn.Linear(nembed, nembed, bias=False)

        self.n_head = nhead

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        #att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        #y = self.resid_drop(self.proj(y))
        y = self.proj(y)
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, nembed, nhead, hidden_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(nembed, eps=1e-6)
        self.ln2 = nn.LayerNorm(nembed, eps=1e-6)
        self.attn = SelfAttention(nembed=nembed, nhead=nhead, dropout=0.0)
        self.mlp = nn.Sequential(
            nn.Linear(nembed, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, nembed),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# https://buomsoo-kim.github.io/attention/2020/04/21/Attention-mechanism-19.md/
class TransformerNet(nn.Module):
    def __init__(self, num_src_vocab, num_tgt_vocab, embedding_dim, hidden_size, nheads, n_layers, max_src_len, is_cls_token = False):
        super(TransformerNet, self).__init__()
        # additional cls token for predicting logp of current input
        if is_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
            self.is_cls_token = True
            self.max_src_len = max_src_len + 1
        else:
            self.is_cls_token = False
            self.max_src_len = max_src_len
        # embedding layers
        self.embedding = nn.Embedding(num_src_vocab, embedding_dim)

        # positional encoding layers
        self.pe = PositionalEncoding(embedding_dim, dropout = 0.0, max_len = self.max_src_len)

        # encoder/decoder layers
        self.decoder = nn.Sequential(*[Block(nembed=embedding_dim, nhead=nheads, hidden_size=hidden_size) for _ in range(n_layers)])
        self.ln_f = LayerNorm(embedding_dim, eps=1e-6)

        # final dense layer or mlp
        # self.dense = nn.Linear(embedding_dim, num_tgt_vocab)

        # self.dense_cls = nn.Linear(embedding_dim, 1)
        self.dense = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, num_tgt_vocab),
        )
        self.dense_cls = nn.Sequential(
            # nn.Linear(embedding_dim*max_src_len, hidden_size),
            nn.Linear(embedding_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            #print("Embedding", module)
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)
        elif isinstance(module, nn.Linear):
            #print("Linear", module)
            torch.nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.Linear) and module.bias is not None:
                #print("Linear bias", module)
                torch.nn.init.normal_(module.bias, mean=0.0, std=1e-6)
        elif isinstance(module, nn.LayerNorm):
            #print("LayerNorm", module)
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x):
        x = x.long()
        y = self.embedding(x)
        # add cls token embedding
        if self.is_cls_token:
            y = torch.cat([self.cls_token.expand(y.size(0), -1, -1), y], dim=1)
        y = self.pe(y)
        
        y = self.decoder(y)
        y = self.ln_f(y)
        # remove cls token embedding
        # y = y[:, 1:, :]
        
        if self.is_cls_token:
            y_cls = y[:, 0, :]
            y = y[:, 1:, :]
        else:
            y_cls = y.mean(dim=1)
            # y_cls = y.flatten(-2, -1)

        # calculate logp of current input
        logp_cls = self.dense_cls(y_cls).squeeze(-1) # (batch_size,)
        logits = self.dense(y)
        return logp_cls, logits # (batch_size,) (batch_size, seq_len, num_tgt_vocab)
    
class TransformerNetDual(nn.Module):
    def __init__(self, num_src_vocab, num_tgt_vocab, embedding_dim, hidden_size, nheads, n_layers, max_src_len, is_cls_token = False):
        super(TransformerNetDual, self).__init__()
        self.net_cd = TransformerNet(
            num_src_vocab, num_tgt_vocab, embedding_dim, hidden_size, nheads, n_layers, max_src_len, is_cls_token)
        self.net_marg = TransformerNet(
            num_src_vocab, num_tgt_vocab, embedding_dim, hidden_size, nheads, n_layers, max_src_len, is_cls_token)
    def forward(self, x):
        logp, _ = self.net_marg(x)
        _, logits = self.net_cd(x)
        return logp, logits # (batch_size,) (batch_size, seq_len, num_tgt_vocab)
    

class TransformerMLPDual(nn.Module):
    def __init__(self, mlp_layers, K, num_src_vocab, num_tgt_vocab, embedding_dim, hidden_size, nheads, n_layers, max_src_len, is_cls_token = False):
        super(TransformerMLPDual, self).__init__()
        self.net_cd = TransformerNet(
            num_src_vocab, num_tgt_vocab, embedding_dim, hidden_size, nheads, n_layers, max_src_len, is_cls_token)
        self.net_marg = MLP(mlp_layers, K)
    def forward(self, x):
        logp = self.net_marg(x)
        _, logits = self.net_cd(x)
        return logp, logits # (batch_size,) (batch_size, seq_len, num_tgt_vocab)