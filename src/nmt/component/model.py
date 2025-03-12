import math
import torch
import torch.nn as nn
from src.nmt.config.configuration import ConfigurationManager


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length):
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_encoding", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.positional_encoding[:, : x.size(1)].to(x.device)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(p=0.1)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(self.dropout(probs), V)

    def split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(
            1, 2
        )

    def combine_heads(self, x):
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, q, k, v, mask=None):
        Q, K, V = (
            self.split_heads(self.w_q(q)),
            self.split_heads(self.w_k(k)),
            self.split_heads(self.w_v(v)),
        )
        context = self.scaled_dot_product_attention(Q, K, V, mask)
        return self.w_o(self.combine_heads(context))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.5):
        super().__init__()
        self.attention = MultiheadAttention(d_model, num_heads)
        self.ffnn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.attention(x, x, x, mask)))
        return self.norm2(x + self.dropout(self.ffnn(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.5):
        super().__init__()
        self.self_attention = MultiheadAttention(d_model, num_heads)
        self.cross_attention = MultiheadAttention(d_model, num_heads)
        self.ffnn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.dropout(self.self_attention(x, x, x, tgt_mask)))
        x = self.norm2(
            x
            + self.dropout(
                self.cross_attention(x, encoder_output, encoder_output, src_mask)
            )
        )
        return self.norm3(x + self.dropout(self.ffnn(x)))


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        max_length,
        num_heads,
        num_layers,
        d_ff,
        dropout=0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_length)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.positional_encoding(self.dropout(self.embedding(src)))
        tgt = self.positional_encoding(self.dropout(self.embedding(tgt)))

        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        for layer in self.decoder_layers:
            tgt = layer(tgt, src, src_mask, tgt_mask)

        return self.fc_out(tgt)


class Model(Transformer):
    def __init__(self, config):
        super().__init__(
            vocab_size=config.model_params.vocab_size,
            d_model=config.model_params.d_model,
            max_length=config.model_params.max_length,
            num_heads=config.model_params.num_heads,
            num_layers=config.model_params.num_layers,
            d_ff=config.model_params.d_ff,
            dropout=config.model_params.dropout,
        )


if __name__ == "__main__":
    try:
        config = ConfigurationManager().get_model_config()
        model = Model(config)
        print("Model initialized successfully.")
    except Exception as e:
        raise RuntimeError("Model initialization failed.") from e
