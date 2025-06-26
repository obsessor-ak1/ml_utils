import torch
from torch import nn
import torch.nn.functional as F


class LanguageModel(nn.Module):
    """A normal class which represents a language model."""
    def __init__(self, rnn, num_vocab, enc_one_hot=True):
        super().__init__()
        self.rnn = rnn
        self.vocab_size = num_vocab
        self.final = nn.LazyLinear(num_vocab)
        self.enc_one_hot = enc_one_hot
    
    def one_hot(self, X):
        """One-hot encodes the input"""
        # Input comes as: (batch_size, time_steps)
        # for RNN, the required shape: (time_steps, batch_size, num_inputs)
        return F.one_hot(X.T, num_classes=self.vocab_size).type(torch.float32)
    
    def forward(self, X):
        embedding = self.one_hot(X) if self.enc_one_hot else X
        rnn_outputs, _ = self.rnn(embedding)
        final_output = self.final(rnn_outputs)
        return final_output.transpose(0, 1)
