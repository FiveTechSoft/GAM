"""
Training / evaluation runner shared across benchmarks.
"""
import numpy as np
import torch
import torch.nn.functional as F

from tasks_data import pad_batch


def run_epoch(model, opt, data_x, data_y, pad_id, device, batch, train=True):
    model.train() if train else model.eval()
    total_loss, n_batches = 0.0, 0
    idx = list(range(len(data_x)))
    if train:
        np.random.shuffle(idx)
    with torch.set_grad_enabled(train):
        for start in range(0, len(idx), batch):
            bi = idx[start:start + batch]
            inp, tgt = pad_batch([data_x[i] for i in bi],
                                 [data_y[i] for i in bi], pad_id, device)
            logits = model(inp)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   tgt.view(-1), ignore_index=-100)
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def accuracy(model, data_x, data_y, pad_id, device, batch):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for start in range(0, len(data_x), batch):
            inp, tgt = pad_batch(data_x[start:start + batch],
                                 data_y[start:start + batch], pad_id, device)
            preds = model(inp).argmax(dim=-1)
            mask = tgt != -100
            correct += (preds[mask] == tgt[mask]).sum().item()
            total += mask.sum().item()
    return correct / max(total, 1)


def train_and_eval(model_factory, make_data, pad_id,
                   *, seeds, epochs, batch, lr, train_n, test_n, device):
    """
    Run model_factory across multiple seeds, return list of test accuracies.

    model_factory(): -> nn.Module (already on device).
    make_data(n, rng): -> (xs, ys).
    """
    accs = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        train_x, train_y = make_data(train_n, rng)
        test_x, test_y = make_data(test_n, rng)

        model = model_factory()
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        for _ in range(epochs):
            run_epoch(model, opt, train_x, train_y, pad_id, device, batch, train=True)
        accs.append(accuracy(model, test_x, test_y, pad_id, device, batch))
    return accs


def summarize(accs):
    a = np.array(accs)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0
