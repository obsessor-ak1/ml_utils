import collections as cls
import math
import random

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F


def cross_corr_2d(X: torch.Tensor, kernel: torch.Tensor, in_channeled=False, out_channeled=False):
    if not in_channeled:
        kernel = kernel[torch.newaxis, :]
        # Assumes image is also in shape: XxY
        X = X[torch.newaxis, :]
    
    if not out_channeled:
        kernel = kernel[torch.newaxis, :]

    kh, kw = kernel.shape[2:]
    ih, iw = X.shape[1:]
    outputs = []
    # For every output channel
    for k in kernel:
        output_ts = torch.zeros((
            ih-kh+1, iw-kw+1
        ))
        for i in range(0, ih-kh+1):
            for j in range(0, iw-kw+1):
                mult = k * X[:, i:i+kh, j:j+kw]
                output_ts[i, j] = mult.sum()
        outputs.append(output_ts)
    
    return torch.stack(outputs)


def prepare_module(module: torch.nn.Module, shape, summarize=True):
    """Passes a dummy tensor through various layers of the module, in order to
    initialize the no. of their input features."""
    X = torch.randn(*shape)
    for child in module.children():
        X = child(X)
        if summarize:
            print(f"{type(child).__name__} output shape: {X.shape}")


def init_seq2seq(module):
    """Initializes weights for sequence to sequence learning."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
    if isinstance(module, nn.GRU):
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])


def masked_softmax(X, valid_lens):
    """Performs softmax operation by masking elements on the last axis."""
    # X: a 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        max_len = X.size(1)
        all_elements = torch.arange(max_len, dtype=torch.float32, device=X.device)
        mask = all_elements[torch.newaxis, :] < valid_len[:, torch.newaxis]
        X[~mask] = value
        return X
    # Performs basic softmax if no valid_lens are provided
    if valid_lens is None:
        return F.softmax(X, dim=-1)
    shape = X.shape
    if valid_lens.dim() == 1:
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
        valid_lens = valid_lens.reshape(-1)
    X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
    return F.softmax(X.reshape(shape), dim=-1)


def get_masked_entropy_loss(target_padding):
    def masked_entropy_loss(Y_hat, Y, reduction="mean"):
        losses = F.cross_entropy(Y_hat, Y, reduction="none")
        mask = (Y != target_padding).type(torch.float32)
        final_losses = losses * mask
        total_loss = final_losses.sum()
        if reduction == "mean":
            return total_loss / mask.sum()
        return total_loss
    return masked_entropy_loss


def bleu(pred_seq, label_seq, k):
    """Compute the BLEU."""
    pred_seq = list(set(pred_seq)-set(["<bos>", "<eos>", "<pad>"]))
    len_label, len_pred = len(label_seq), len(pred_seq)
    score = math.exp(min(0, 1- len_label / len_pred))
    for n in range(1, min(k, len_pred+1)):
        num_matches, label_subs = 0, cls.defaultdict(int)
        for i in range(len_label-n-1):
            label_subs[''.join(label_seq[i:i+n])] += 1
        for i in range(len_pred-n+1):
            if label_subs[''.join(pred_seq[i:i+n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_seq[i:i+n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
        return score


def show_dataset_images(dataset, nrows=4, ncols=4):
    ds_count = len(dataset)
    image_ids = random.sample(range(ds_count), k=nrows*ncols)
    image_ids = torch.tensor(image_ids).reshape(nrows, ncols)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for i in range(nrows):
        for j in range(ncols):
            image, label = dataset[image_ids[i][j]]
            class_title = dataset.classes[label.item()]
            axes[i][j].imshow(image.permute(1, 2, 0).numpy())
            axes[i][j].axis("off")
            axes[i][j].set_title(class_title)