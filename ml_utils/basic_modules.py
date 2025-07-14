import math

import torch
from torch import nn
import torch.nn.functional as F

from ml_utils.basic_utils import masked_softmax


class LanguageModel(nn.Module):
    """A normal class which represents a language model."""
    def __init__(self, rnn, num_vocab, embedding_layer=None):
        super().__init__()
        self.rnn = rnn
        self.vocab_size = num_vocab
        self.final = nn.LazyLinear(num_vocab)
        self.embedding_layer = embedding_layer
    
    def one_hot(self, X):
        """One-hot encodes the input"""
        # Input comes as: (batch_size, time_steps)
        # for RNN, the required shape: (time_steps, batch_size, num_inputs)
        return F.one_hot(X.T, num_classes=self.vocab_size).type(torch.float32)

    def encode(self, X):
        if self.embedding_layer:
            return self.embedding_layer(X).transpose(0, 1)
        return self.one_hot(X)
    
    def forward(self, X):
        embedding = self.encode(X)
        rnn_outputs, _ = self.rnn(embedding)
        final_output = self.final(rnn_outputs)
        return final_output.transpose(0, 1)

    def predict_text(self, prefix, num_preds, tokenizer, vocab, device="cpu"):
        # Batchifying the text
        self.eval()
        prefix_tokens = torch.tensor([vocab[tokenizer(prefix)]], device=device)
        prefix_tokens = self.encode(prefix_tokens)
        # Warm-up
        output, state = self.rnn(prefix_tokens)
        # Prediction
        first_pred = self.final(output)[-1, 0].argmax()
        outputs = [first_pred]
        for i in range(num_preds-1):
            X = torch.tensor([[outputs[-1]]], device=device)
            emb = self.encode(X)
            output, state = self.rnn(emb, state)
            pred = self.final(output[-1, 0]).argmax()
            outputs.append(pred)
        suffix = ''.join(vocab.to_tokens(outputs))
        return prefix+suffix


class AdditiveAttention(nn.Module):
    """Additive attention implementation."""
    def __init__(self, num_hidden, dropout):
        super().__init__()
        self.W_q = nn.LazyLinear(num_hidden, bias=False)
        self.W_k = nn.LazyLinear(num_hidden, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # print(f"Queries shape: {queries.shape}, Keys shape: {keys.shape}")
        # print(f"Features shape: {features.shape}")
        scores = self.w_v(features).squeeze(-1)
        # print(f"Scores shape: {scores.shape}")
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    """A simple dot product attention (scaled) implementation."""
    def __init__(self, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.size(-1)
        # Swapping the last two dimenstions of the keys, to support dot product
        # using matrix multiplication
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    """Multi-head implementation."""
    def __init__(self, num_hidden, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(num_hidden, bias=bias)
        self.W_k = nn.LazyLinear(num_hidden, bias=bias)
        self.W_v = nn.LazyLinear(num_hidden, bias=bias)
        self.W_o = nn.LazyLinear(num_hidden, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))
        if valid_lens is not None:
            valid_lens = valid_lens.repeat_interleave(
                repeats=self.num_heads, dim=0)
        output = self.attention(queries, keys, values, valid_lens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)

    def transpose_qkv(self, X):
        """Transposition for parallel computation of multiple attention heads."""
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X):
        """Reverse the operation of transpose_qkv."""
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

    @property
    def attention_weights(self):
        return self.attention.attention_weights


class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, num_hidden, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hidden))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(0, num_hidden, 2, dtype=torch.float32) / num_hidden)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)