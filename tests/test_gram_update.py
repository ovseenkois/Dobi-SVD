import torch
from torch import nn

from modules.gram_update import collect_gram_matrices, gram_project_weight
from modules.module import ProjectedLinear


def test_gram_projection_matches_best_rank_k_calibration_output():
    torch.manual_seed(7)
    linear = nn.Linear(5, 4, bias=True)
    inputs = torch.randn(12, 5)
    original_output = linear(inputs)
    gram = (original_output - linear.bias).T @ (original_output - linear.bias)

    basis, projected = gram_project_weight(gram, linear.weight, rank=2)
    compressed = ProjectedLinear(nn.Linear(5, 4, bias=True), basis)
    with torch.no_grad():
        compressed.ALinear.weight.copy_(basis.T @ projected)
        compressed.BLinear.weight.copy_(basis)
        compressed.BLinear.bias.copy_(linear.bias)

    actual = compressed(inputs) - linear.bias
    expected = (original_output - linear.bias) @ basis @ basis.T
    assert torch.allclose(actual, expected, atol=1e-5)

    singular_values = torch.linalg.svdvals(original_output - linear.bias)
    residual = torch.linalg.norm((original_output - linear.bias) - actual) ** 2
    assert torch.allclose(residual, singular_values[2:].square().sum(), atol=1e-4)


def test_collection_uses_original_outputs_without_changing_forward():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(3, 2, bias=True)

        def forward(self, input_ids):
            return self.proj(input_ids.float())

    model = TinyModel()
    samples = [{'input_ids': torch.tensor([[1, 2, 3], [2, 0, 1]])},
               {'input_ids': torch.tensor([[0, 1, 1]])}]
    full_input = torch.cat([sample['input_ids'] for sample in samples]).float()
    outputs_before = model(full_input)
    grams = collect_gram_matrices(model, ['proj'], samples, torch.device('cpu'))

    linear_output = outputs_before - model.proj.bias
    assert torch.allclose(grams['proj'], linear_output.T @ linear_output)
    assert torch.allclose(model(full_input), outputs_before)
    assert not model.proj._forward_hooks
