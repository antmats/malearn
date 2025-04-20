from sklearn.tree._tree import TREE_UNDEFINED, TREE_LEAF

from malearn.estimators.tree import update_tree_structure


class TreeNode:
    def __init__(self, feature, threshold, left_subtree=None, right_subtree=None):
        self.feature = feature
        self.threshold = threshold
        self.left_subtree = left_subtree
        self.right_subtree = right_subtree


def test_no_pruning():
    """Test when no pruning is applied, the tree remains unchanged."""
    root = TreeNode(
        feature=0,
        threshold=1.5,
        left_subtree=TreeNode(TREE_UNDEFINED, TREE_UNDEFINED),
        right_subtree=TreeNode(TREE_UNDEFINED, TREE_UNDEFINED),
    )

    nodes_after_pruning = [
        (1, 2),  # Root points to left=1, right=2
        (TREE_LEAF, TREE_LEAF),  # Left child remains leaf
        (TREE_LEAF, TREE_LEAF),  # Right child remains leaf
    ]

    expected_tree = TreeNode(
        feature=0,
        threshold=1.5,
        left_subtree=TreeNode(TREE_UNDEFINED, TREE_UNDEFINED),
        right_subtree=TreeNode(TREE_UNDEFINED, TREE_UNDEFINED),
    )

    update_tree_structure(root, nodes_after_pruning[0], nodes_after_pruning)

    assert root.feature == expected_tree.feature
    assert root.left_subtree.feature == expected_tree.left_subtree.feature
    assert root.right_subtree.feature == expected_tree.right_subtree.feature


def test_full_pruning():
    """Test when all nodes become leaves after pruning."""
    root = TreeNode(
        feature=0,
        threshold=1.5,
        left_subtree=TreeNode(TREE_UNDEFINED, TREE_UNDEFINED),
        right_subtree=TreeNode(TREE_UNDEFINED, TREE_UNDEFINED),
    )

    nodes_after_pruning = [
        (TREE_LEAF, TREE_LEAF)  # Root becomes a leaf
    ]

    update_tree_structure(root, nodes_after_pruning[0], nodes_after_pruning)

    assert root.feature == TREE_UNDEFINED
    assert root.threshold == TREE_UNDEFINED
    assert root.left_subtree is None
    assert root.right_subtree is None


def test_internal_node_becomes_leaf():
    """Test when an internal node loses both children and becomes a leaf."""
    root = TreeNode(
        feature=0,
        threshold=1.5,
        left_subtree=TreeNode(
            feature=1,
            threshold=2.0,
            left_subtree=TreeNode(TREE_UNDEFINED, TREE_UNDEFINED),
            right_subtree=TreeNode(TREE_UNDEFINED, TREE_UNDEFINED),
        ),
        right_subtree=TreeNode(TREE_UNDEFINED, TREE_UNDEFINED),
    )

    nodes_after_pruning = [
        (1, 2),  # Root points to left=1, right=2
        (TREE_LEAF, TREE_LEAF),  # Left subtree is pruned
        (TREE_LEAF, TREE_LEAF),  # Right child remains leaf
    ]

    expected_tree = TreeNode(
        feature=0,
        threshold=1.5,
        left_subtree=TreeNode(TREE_UNDEFINED, TREE_UNDEFINED),
        right_subtree=TreeNode(TREE_UNDEFINED, TREE_UNDEFINED),
    )

    update_tree_structure(root, nodes_after_pruning[0], nodes_after_pruning)

    assert root.feature == expected_tree.feature
    assert root.left_subtree.feature == expected_tree.left_subtree.feature
    assert root.right_subtree.feature == expected_tree.right_subtree.feature
