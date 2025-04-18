import math
import copy

import numpy as np
from sklearn.tree import _tree


class Node:
    def __init__(
        self,
        index,
        parent=None,
        left_child=None,
        right_child=None,
    ):
        self.index = index
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child
    
    def add_children(self):
        self.left_child = Node(index=2 * self.index + 1, parent=self)
        self.right_child = Node(index=2 * self.index + 2, parent=self)

    def remove_children(self):
        self.left_child = None
        self.right_child = None
    
    @property
    def layer_index(self):
        return int(math.log2(self.index + 1))
    
    @property
    def is_leaf(self):
        return self.left_child is None and self.right_child is None


def count_nodes(node):
    if node is not None:
        return (
            1 + count_nodes(node.left_subtree) + count_nodes(node.right_subtree)
        )
    return 0


class TreeNode:
    def __init__(
        self,
        value,
        impurity,
        *,
        n_node_samples,
        weighted_n_node_samples,
        depth,
        feature=None,
        threshold=None,
        left_subtree=None,
        right_subtree=None,
        missingness_reliance=None,
    ):
        self.value = value
        self.impurity = impurity

        self.n_node_samples = n_node_samples
        self.weighted_n_node_samples = weighted_n_node_samples

        # We do not handle missing values in this implementation.
        self.missing_go_to_left = 0

        self.depth = depth

        undefined = _tree.TREE_UNDEFINED
        self.feature = feature if feature is not None else undefined
        self.threshold = threshold if threshold is not None else undefined

        self.left_subtree = left_subtree
        self.right_subtree = right_subtree

        self.missingness_reliance = missingness_reliance

    def predict(self, x, m=None, return_depth=False):
        if self.feature == _tree.TREE_UNDEFINED:
            if return_depth:
                return self.value, self.depth
            return self.value
        if m is not None and m[self.feature] == 1:
            if return_depth:
                return self.value, self.depth
            return self.value
        if x[self.feature] <= self.threshold:
            return self.left_subtree.predict(x, m, return_depth)
        else:
            return self.right_subtree.predict(x, m, return_depth)

    @property
    def is_leaf(self):
        return self.feature == _tree.TREE_UNDEFINED

    def __eq__(self, other):
        return (
            isinstance(other, TreeNode)
            # np.array_equal works for both arrays and scalars.
            and np.array_equal(self.value, other.value)
        )


def _populate_node_list(node, node_list):
    node = copy.deepcopy(node)

    if node.feature == _tree.TREE_UNDEFINED:
        node.left_child = _tree.TREE_LEAF
        node.right_child = _tree.TREE_LEAF

        node_list.append(node)
    else:
        left_idx = len(node_list) + 1
        right_idx = left_idx + count_nodes(node.left_subtree)

        node.left_child = left_idx
        node.right_child = right_idx

        node_list.append(node)

        _populate_node_list(node.left_subtree, node_list)
        _populate_node_list(node.right_subtree, node_list)


def convert_to_sklearn_tree(root_node, n_features, n_classes):
    nodes = []
    _populate_node_list(root_node, nodes)

    max_depth = max(node.depth for node in nodes)
    node_count = len(nodes)

    values = [node.value for node in nodes]
    values = np.array(values).reshape(-1, 1, n_classes)

    nodes = [
        (
            node.left_child,
            node.right_child,
            node.feature,
            node.threshold,
            node.impurity,
            node.n_node_samples,
            node.weighted_n_node_samples,
            node.missing_go_to_left,
        )
        for node in nodes
    ]
    nodes = np.array(nodes, dtype=_tree.NODE_DTYPE)

    tree = _tree.Tree(n_features, np.atleast_1d(n_classes), 1)
    tree.__setstate__(
        {
            "max_depth": max_depth,
            "node_count": node_count,
            "nodes": nodes,
            "values": values,
        }
    )

    return tree


class ExtendedTree:
    def __init__(self, original_tree, root_node):
        self.n_features = original_tree.n_features
        self.n_classes = original_tree.n_classes
        self.n_outputs = original_tree.n_outputs
        state = original_tree.__getstate__()
        state["node_array"] = state.pop("nodes")
        state["value_array"] = state.pop("values")
        self.__dict__.update(state)
        self.nodes = list(self._yield_nodes(root_node))

    @property
    def children_left(self):
        return self.node_array["left_child"][:self.node_count]

    @property
    def children_right(self):
        return self.node_array["right_child"][:self.node_count]

    @property
    def feature(self):
        return self.node_array["feature"][:self.node_count]

    @property
    def threshold(self):
        return self.node_array["threshold"][:self.node_count]

    @property
    def impurity(self):
        return self.node_array["impurity"][:self.node_count]

    @property
    def n_node_samples(self):
        return self.node_array["n_node_samples"][:self.node_count]

    @property
    def weighted_n_node_samples(self):
        return self.node_array["weighted_n_node_samples"][:self.node_count]

    @property
    def missing_go_to_left(self):
        return self.node_array["missing_go_to_left"][:self.node_count]

    @property
    def value(self):
        return self.value_array[:self.node_count]

    @property
    def missingness_reliance(self):
        return [node.missingness_reliance for node in self.nodes]

    def _yield_nodes(self, root_node):
        node_stack = [root_node]
        while node_stack:
            node = node_stack.pop()
            yield node
            if node.right_subtree is not None:
                node_stack.append(node.right_subtree)
            if node.left_subtree is not None:
                node_stack.append(node.left_subtree)

    def apply(self, X):
        out = []
        n_samples = X.shape[0]
        indexed_nodes = [(i, *node) for i, node in enumerate(self.node_array)]
        for i in range(n_samples):
            node = indexed_nodes[0]  # Start at the root
            while node[1] != _tree.TREE_LEAF:
                if X[i, node[3]] <= node[4]:
                    node = indexed_nodes[node[1]]
                else:
                    node = indexed_nodes[node[2]]
            out.append(node[0])
        return np.array(out)


def update_tree_structure(old_node, new_node, nodes_after_pruning):
    if new_node[0] == _tree.TREE_LEAF and new_node[1] == _tree.TREE_LEAF:
        old_node.feature = old_node.threshold = _tree.TREE_UNDEFINED
        old_node.left_child = old_node.right_child = _tree.TREE_LEAF
        old_node.left_subtree = old_node.right_subtree = None
        return
    else:
        assert new_node[0] != _tree.TREE_LEAF or new_node[1] != _tree.TREE_LEAF
        update_tree_structure(
            old_node.left_subtree,
            nodes_after_pruning[new_node[0]],
            nodes_after_pruning,
        )
        update_tree_structure(
            old_node.right_subtree,
            nodes_after_pruning[new_node[1]],
            nodes_after_pruning,
        )
