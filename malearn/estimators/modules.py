import torch
from torch import Tensor, nn
from torch.types import _dtype
from torch.utils.data.dataloader import default_collate

import numpy as np


# TODO: Move this function to a more appropriate location.
def collate_fn(batch):
    def process_item(item):
        if isinstance(item, np.ndarray) and item.dtype == "object":
            return np.array(item, dtype=np.float32)
        return item
    return default_collate([(process_item(X), y) for X, y in batch])


# Source: https://github.com/marineLM/NeuMiss_sota/blob/master/neumiss/NeuMissBlock.py#L7
class Mask(nn.Module):
    """A mask non-linearity."""
    mask: Tensor

    def __init__(self, input: Tensor):
        super(Mask, self).__init__()
        self.mask = torch.isnan(input)

    def forward(self, input: Tensor) -> Tensor:
        return ~self.mask*input


# Source: https://github.com/marineLM/NeuMiss_sota/blob/master/neumiss/NeuMissBlock.py#L19
class SkipConnection(nn.Module):
    """A skip connection operation."""
    value: Tensor

    def __init__(self, value: Tensor):
        super(SkipConnection, self).__init__()
        self.value = value

    def forward(self, input: Tensor) -> Tensor:
        return input + self.value


# Source: https://github.com/marineLM/NeuMiss_sota/blob/master/neumiss/NeuMissBlock.py#L31
class NeuMissBlock(nn.Module):
    """The NeuMiss block from "What’s a good imputation to predict with
    missing values?" by Marine Le Morvan, Julie Josse, Erwan Scornet,
    Gael Varoquaux.
    """

    def __init__(self, n_features: int, depth: int,
                 dtype: _dtype = torch.float) -> None:
        """
        Parameters
        ----------
        n_features : int
            Dimension of inputs and outputs of the NeuMiss block.
        depth : int
            Number of layers (Neumann iterations) in the NeuMiss block.
        dtype : _dtype
            Pytorch dtype for the parameters. Default: torch.float.

        """
        super().__init__()
        self.depth = depth
        self.dtype = dtype
        self.mu = nn.Parameter(torch.empty(n_features, dtype=dtype))
        self.linear = nn.Linear(n_features, n_features, bias=False, dtype=dtype)
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = x.type(self.dtype)  # Cast tensor to appropriate dtype
        mask = Mask(x)  # Initialize mask non-linearity
        x = torch.nan_to_num(x)  # Fill missing values with 0
        h = x - mask(self.mu)  # Subtract masked parameter mu
        skip = SkipConnection(h)  # Initialize skip connection with this value

        layer = [self.linear, mask, skip]  # One Neumann iteration
        layers = nn.Sequential(*(layer*self.depth))  # Neumann block

        return layers(h)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.mu)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.5)

    def extra_repr(self) -> str:
        return "depth={}".format(self.depth)


# Source: https://github.com/marineLM/NeuMiss_sota/blob/master/neumiss/NeuMissBlock.py#L76
class NeuMissMLP(nn.Module):
    """A NeuMiss block followed by an MLP.

    Parameters
    ----------
    n_features : int
        Dimension of inputs.
    output_dim : int
        Dimension of outputs.
    neumiss_depth : int
        Number of layers in the NeuMiss block.
    mlp_depth : int
        Number of hidden layers in the MLP.
    mlp_width : int
        Width of the MLP. If None take mlp_width=n_features. Default: None.
    dtype : _dtype
        Pytorch dtype for the parameters. Default: torch.float.
    """

    def __init__(
        self,
        n_features: int,
        output_dim: int,
        neumiss_depth: int,
        mlp_depth: int,
        mlp_width: int = None,
        dtype: _dtype = torch.float,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.output_dim = output_dim
        self.neumiss_depth = neumiss_depth
        self.mlp_depth = mlp_depth
        mlp_width = n_features if mlp_width is None else mlp_width
        self.mlp_width = mlp_width
        self.dtype = dtype

        self.layers = nn.Sequential(
            NeuMissBlock(self.n_features, self.neumiss_depth, self.dtype),
            *self._build_mlp_layers(
                self.n_features,
                self.output_dim,
                self.mlp_width,
                self.mlp_depth,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

    def _build_mlp_layers(self, input_dim, output_dim, width, depth):
        """Build the MLP layers with the specified depth and width."""
        layers = []
        if depth >= 1:
            layers += [nn.Linear(input_dim, width, dtype=self.dtype), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width, dtype=self.dtype), nn.ReLU()]
        last_layer_width = width if depth >= 1 else self.n_features
        layers += [nn.Linear(last_layer_width, output_dim, dtype=self.dtype)]
        return layers
