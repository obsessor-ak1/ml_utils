import torch


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


def predict_text(model, prefix, num_preds, vocab, device=None):
    state, outputs = None, [vocab[prefix[0]]]
    for i in range(len(prefix) + num_preds -1):
        X = torch.tensor([[outputs[-1]]], device=device)
        embs = model.one_hot(X)
        rnn_outputs, state = model.rnn(embs, state)
        if i < len(prefix)-1:
            outputs.append(vocab[prefix[i+1]])
        else:
            Y = model.output_layer(rnn_outputs)
            outputs.append(
                int(Y.argmax(axis=2).reshape(1))
            )
    return ''.join([vocab.idx_to_token[i] for i in outputs])