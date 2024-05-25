import torch


def geometric_crossover(tree1, tree2, random_tree, testing):

    if testing:
        return torch.add(
            torch.mul(tree1.test_semantics, random_tree.test_semantics),
            torch.mul(torch.sub(1, random_tree.test_semantics), tree2.test_semantics),
        )

    else:
        return torch.add(
            torch.mul(tree1.train_semantics, random_tree.train_semantics),
            torch.mul(torch.sub(1, random_tree.train_semantics), tree2.train_semantics),
        )
