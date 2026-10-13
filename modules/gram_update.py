"""Activation Gram statistics and exact output-space weight projection."""

import torch
from torch import nn
from tqdm import tqdm


def collect_gram_matrices(model, names, batches, device):
    """Return sum A.T @ A for each named Linear output, with A from the original model.

    Accumulators live in host RAM so that the GPU only needs one temporary Gram
    matrix at a time. Every token contributes once; PCA centering is not applied.
    """
    modules = dict(model.named_modules())
    grams = {
        name: torch.zeros((modules[name].out_features, modules[name].out_features),
                          dtype=torch.float32, device='cpu')
        for name in names
    }
    handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            activations = output.detach().reshape(-1, module.out_features).float()
            if module.bias is not None:
                activations = activations - module.bias.detach().float()
            grams[name].add_((activations.T @ activations).cpu())
        return hook

    try:
        for name in names:
            handles.append(modules[name].register_forward_hook(make_hook(name)))
        model.eval()
        with torch.no_grad():
            for batch in tqdm(batches, desc='Collect activation Gram matrices'):
                inputs = {key: value.to(device).unsqueeze(0) for key, value in batch.items()}
                model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    return grams


def gram_project_weight(gram, weight, rank):
    """Project [out, in] weights onto the top output directions of the Gram.

    Returns an orthonormal [out, rank] basis and projected [out, in] weight.
    """
    out_features, in_features = weight.shape
    if gram.shape != (out_features, out_features):
        raise ValueError(f'Gram shape {tuple(gram.shape)} does not match output size {out_features}')
    if not 1 <= rank <= min(out_features, in_features):
        raise ValueError(f'Invalid rank {rank} for weight {tuple(weight.shape)}')

    work_device = weight.device
    symmetric = ((gram + gram.T) * 0.5).to(work_device)
    _, vectors = torch.linalg.eigh(symmetric)
    basis = vectors[:, -rank:].contiguous()
    weight_float = weight.to(torch.float32)
    projected = basis @ (basis.T @ weight_float)
    return basis, projected
