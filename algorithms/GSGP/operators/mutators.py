import torch
import random


def standard_geometric_mutation(tree, random_tree_1, random_tree_2, ms, testing):

    if testing:
        return torch.add(
            tree.test_semantics,
            torch.mul(
                ms,
                torch.sub(random_tree_1.test_semantics, random_tree_2.test_semantics),
            ),
        )

    else:
        return torch.add(
            tree.train_semantics,
            torch.mul(
                ms,
                torch.sub(random_tree_1.train_semantics, random_tree_2.train_semantics),
            ),
        )


def standard_one_tree_geometric_mutation(tree, random_tree_1, ms, testing):

    if testing:
        return torch.add(
            tree.test_semantics,
            torch.mul(
                ms,
                torch.sub(
                    1,
                    torch.div(2, torch.add(1, torch.abs(random_tree_1.test_semantics))),
                ),
            ),
        )

    else:
        return torch.add(
            tree.train_semantics,
            torch.mul(
                ms,
                torch.sub(
                    1,
                    torch.div(
                        2, torch.add(1, torch.abs(random_tree_1.train_semantics))
                    ),
                ),
            ),
        )


def product_two_trees_geometric_mutation(
    tree, random_tree_1, random_tree_2, ms, testing
):

    if testing:
        return torch.mul(
            tree.test_semantics,
            torch.add(
                1,
                torch.mul(
                    ms,
                    torch.sub(
                        random_tree_1.test_semantics, random_tree_2.test_semantics
                    ),
                ),
            ),
        )

    else:
        return torch.mul(
            tree.train_semantics,
            torch.add(
                1,
                torch.mul(
                    ms,
                    torch.sub(
                        random_tree_1.train_semantics, random_tree_2.train_semantics
                    ),
                ),
            ),
        )


def product_one_trees_geometric_mutation(tree, random_tree_1, ms, testing):

    if testing:
        return torch.mul(
            tree.test_semantics,
            torch.add(
                1,
                torch.mul(
                    ms,
                    torch.sub(
                        1,
                        torch.div(
                            2, torch.add(1, torch.abs(random_tree_1.test_semantics))
                        ),
                    ),
                ),
            ),
        )

    else:
        return torch.mul(
            tree.train_semantics,
            torch.add(
                1,
                torch.mul(
                    ms,
                    torch.sub(
                        1,
                        torch.div(
                            2, torch.add(1, torch.abs(random_tree_1.train_semantics))
                        ),
                    ),
                ),
            ),
        )
