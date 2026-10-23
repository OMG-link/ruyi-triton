import pytest
import torch

from .scatter_reduce import scatter_reduce, scatter_reduce_


@pytest.mark.parametrize(
    "shape, dim, reduce",
    [
        ((4, 8), 0, "sum"),
        ((4, 8), 1, "sum"),
    ],
)
def test_scatter_reduce(shape, dim, reduce):
    torch.manual_seed(0)
    inp = torch.randn(*shape, dtype=torch.float32)
    src = torch.randn(*shape, dtype=torch.float32)
    index = torch.randint(0, shape[dim], (shape[dim],), dtype=torch.long)
    # For non-dim reduction, index matches src shape
    index_expanded = index
    if dim == 0:
        index_expanded = index.unsqueeze(1).expand_as(src)
    elif dim == 1:
        index_expanded = index.unsqueeze(0).expand_as(src)

    ref = torch.scatter_reduce(inp.clone(), dim, index_expanded, src, reduce=reduce)
    tri = scatter_reduce(inp.clone(), dim, index_expanded, src, reduce=reduce)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dim", [-3, 2])
def test_scatter_reduce_rejects_invalid_dim(dim):
    inp = torch.zeros((2, 2))
    index = torch.zeros((2, 2), dtype=torch.long)
    src = torch.ones((2, 2))

    with pytest.raises(IndexError):
        scatter_reduce(inp, dim, index, src, reduce="sum")
