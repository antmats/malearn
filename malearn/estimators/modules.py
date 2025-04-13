import torch
from torch import Tensor, nn
from torch.types import _dtype
from torch.utils.data.dataloader import default_collate

import numpy as np

from .tree import get_node_indices_per_layer


# TODO: Move this function to a more appropriate location.
def collate_fn(batch):
    def process_item(item):
        if isinstance(item, np.ndarray) and item.dtype == "object":
            return np.array(item, dtype=np.float32)
        return item
    return default_collate([(process_item(X), y) for X, y in batch])


class UnidimensionalInnerNodes(nn.Module):
    def __init__(self, weight):
        super(UnidimensionalInnerNodes, self).__init__()
        self.weight = weight

    def forward(self, x):
        """Perform a forward pass using unidimensional inner nodes.
        See https://arxiv.org/pdf/1903.09338 for details.
        """

        # Split the weights.
        wb = torch.t(self.weight)
        input_dim = self.weight.size(1) - 1
        b, w = torch.split(wb, [1, input_dim], dim=0)

        # Split the input.
        _, x = torch.split(x, [1, input_dim], dim=1)

        # Select the largest weight for each inner node.
        wmax, max_indices = torch.max(w, dim=0)  # Shape (# inner nodes,)

        # Select the observations corresponding to the largest weights.
        xmax = x[:, max_indices]  # Shape (batch size, # inner nodes)

        # Compute the thresholds.
        thresholds = -(b / wmax)

        # For each sample in the batch, compute
        # zmax > -(bh / wmax) if wmax > 0
        # zmax < -(bh / wmax) if wmax < 0
        condition = wmax > 0
        input = xmax > thresholds
        other = xmax < thresholds
        return torch.where(condition, input, other).type(torch.float32)


class SDT(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        tree,
        prediction_mode="max",
    ):
        super(SDT, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.tree = tree
        self.prediction_mode = prediction_mode

        self.penalty_decay_per_layer = [2 ** (-d) for d in range(tree.depth)]

        num_inner_nodes = len(tree.inner_nodes)
        self.inner_nodes = nn.Sequential(
            nn.Linear(
                self.input_dim + 1,  # Add bias
                num_inner_nodes,
                bias=False,
            ),
            nn.Sigmoid(),
        )

        num_leaf_nodes = len(tree.leaf_nodes)
        self.leaf_nodes = nn.Sequential(
            nn.Linear(
                num_leaf_nodes,
                self.output_dim,
                bias=False,
            ),
            nn.Identity(),
        )

    def discretize(self):
        if not isinstance(self.inner_nodes, UnidimensionalInnerNodes):
            inner_node_weights = self.inner_nodes[0].weight
            self.inner_nodes = UnidimensionalInnerNodes(inner_node_weights)

    def forward(self, X, M=None):
        path_proba, splitting_penalty, all_path_proba = self._forward(X)

        if self.prediction_mode == "max":
            masked_path_proba = self._mask_path_proba(path_proba)
            logits = self.leaf_nodes(masked_path_proba)
        else:  # "mean"
            logits = self.leaf_nodes(path_proba)
        
        if M is not None:
            # Compute missingness penalty.
            penalty_decays = [
                penalty
                for depth, penalty in enumerate(self.penalty_decay_per_layer)
                for _ in range(2 ** depth)
            ]
            penalty_decays = torch.tensor(penalty_decays).to(M.device)

            # TODO: Should we take the path probabilities into account here?
            inner_node_weights = self.inner_nodes[0].weight[:, 1:].unsqueeze(0)
            missingness_penalty = M.unsqueeze(1) * torch.abs(inner_node_weights)
            missingness_penalty = torch.sum(missingness_penalty, dim=-1)
            missingness_penalty = missingness_penalty * penalty_decays
            missingness_penalty = torch.sum(missingness_penalty)
        else:
            missingness_penalty = torch.tensor(0.0).to(logits.device)

        return {
            "probas": (path_proba, all_path_proba),
            "penalties": (splitting_penalty, missingness_penalty),
            "predictions": (logits,),
        }

    def _forward(self, X):
        batch_size = X.size(0)

        X = self._data_augment(X)

        path_proba = self.inner_nodes(X)
        path_proba = torch.unsqueeze(path_proba, dim=2)
        path_proba = torch.cat((1 - path_proba, path_proba), dim=2)

        mu = X.data.new(batch_size, 1, 1).fill_(1.0)
        penalty = torch.tensor(0.0).to(mu.device)
        all_path_proba = [mu.view(batch_size, -1)]
        
        node_indices_per_layer = get_node_indices_per_layer(self.tree.depth)
        
        shift = 0
        for i_layer in range(self.tree.depth):
            node_indices_in_full_layer = node_indices_per_layer[i_layer]
            num_nodes_in_full_layer = len(node_indices_in_full_layer)

            nodes_in_layer = self.tree.get_inner_nodes_at_layer(i_layer)
            num_nodes_in_layer = len(nodes_in_layer)
            
            if num_nodes_in_full_layer > num_nodes_in_layer:
                layer_aligned_node_indices = [
                    node.index - shift for node in nodes_in_layer.values()
                ]
                tree_aligned_node_indices = [
                    self.tree.get_inner_node_index(node)
                    for node in nodes_in_layer.values()
                ]
                path_proba_ = torch.full(
                    (batch_size, num_nodes_in_full_layer, 2),
                    1.0,
                    dtype=path_proba.dtype,
                    device=path_proba.device,
                )
                path_proba_[:, layer_aligned_node_indices, :] = \
                    path_proba[:, tree_aligned_node_indices, :]
            else:
                path_proba_ = path_proba[:, node_indices_in_full_layer, :]

            penalty += self._cal_penalty(i_layer, mu, path_proba_)
            
            # `mu` grows from the initial shape (B, 1, 1) to (B, 1, 2) to
            # (B, 2, 2) to (B, 4, 2) to (B, 8, 2), etc.
            #
            # For example, when `i_layer = 1`, the line below does the
            # following transformation:
            # mu = [[[a, b]], [[a, b]]]  (shape (B, 1, 2))
            # mu.view(batch_size, -1, 1) = [[[a], [b]], [[a], [b]]]  (shape (B, 2, 1))
            # mu.view(batch_size, -1, 1).repeat(1, 1, 2) = [[[a, a], [b, b]], [[a, a], [b, b]]]  (shape (B, 2, 2))
            mu = mu.view(batch_size, -1, 1).repeat(1, 1, 2)
            
            # Get the path probabilities for the nodes in the current layer.
            mu = mu * path_proba_
 
            # Store the path probabilities for the nodes in the current layer.
            mu_ = mu.detach().clone()
            if num_nodes_in_full_layer > num_nodes_in_layer:
                mask = torch.ones(num_nodes_in_full_layer, dtype=torch.bool, device=mu.device)
                mask[layer_aligned_node_indices] = False
                mu_[:, mask, :] = np.nan
            mu_ = mu_.view(batch_size, -1)
            all_path_proba += [mu_]

            shift += 2 ** i_layer
    
        mu = mu.view(batch_size, -1)
        assert mu.shape[1] == len(self.tree.leaf_nodes)

        return mu, penalty, torch.cat(all_path_proba, dim=1)

    def _cal_penalty(self, i_layer, mu, path_prob):
        penalty = torch.tensor(0.0).to(mu.device)

        num_nodes_in_layer = 2 ** i_layer
        num_nodes_in_next_layer = 2 ** (i_layer + 1)

        batch_size = mu.size(0)
        mu = mu.view(batch_size, num_nodes_in_layer)
        path_prob = path_prob.view(batch_size, num_nodes_in_next_layer)

        for i_node in range(num_nodes_in_layer):
            node_index = 2 ** i_layer - 1 + i_node

            if not node_index in self.tree.inner_nodes:
                continue
            
            # Find the index of the right child.
            right_child_index = 2 * node_index + 2

            # Convert the the index to be within [0, `num_nodes_in_next_layer`).
            right_child_index -= num_nodes_in_next_layer
            assert 0 <= right_child_index < num_nodes_in_next_layer

            alpha = torch.sum(
                path_prob[:, right_child_index] * mu[:, i_node],
                dim=0,
            )
            alpha /= torch.sum(mu[:, i_node], dim=0)
            penalty_ = (torch.log(alpha) + torch.log(1 - alpha))

            if penalty_.isnan() or penalty_.isinf():
                # TODO: Handle this case.
                continue

            penalty -= 0.5 * self.penalty_decay_per_layer[i_layer] * penalty_

        return penalty
    
    def _data_augment(self, X):
        batch_size = X.size(0)
        X = X.view(batch_size, -1)
        bias = torch.ones(batch_size, 1).to(X.device)
        return torch.cat((bias, X), 1)
    
    def _mask_path_proba(self, path_proba):
        path_proba_np = path_proba.detach().cpu().numpy()
        assert np.allclose(path_proba_np.sum(axis=1), 1.0)
        max_indices = path_proba.argmax(dim=1).unsqueeze(1)
        mask = torch.zeros_like(path_proba)
        mask.scatter_(1, max_indices, 1.0)
        return mask


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
        return 'depth={}'.format(self.depth)


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
