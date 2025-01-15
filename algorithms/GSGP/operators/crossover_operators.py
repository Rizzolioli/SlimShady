import torch

def geometric_crossover(tree1, tree2, random_tree, testing):

    if testing:
        return torch.add(torch.mul(tree1.test_semantics, random_tree.test_semantics), torch.mul(torch.sub(1, random_tree.test_semantics), tree2.test_semantics))

    else:
        return torch.add(torch.mul(tree1.train_semantics, random_tree.train_semantics), torch.mul(torch.sub(1, random_tree.train_semantics), tree2.train_semantics))

def combined_geometric_crossover(tree1, tree2, random_tree_1, random_tree_2, random_tree_3, random_tree_4, random_tree_5, ms, testing):

    if testing:
        return torch.add(torch.mul(
            torch.add(tree1.test_semantics, torch.mul(ms, torch.sub(random_tree_2.test_semantics, random_tree_3.test_semantics))),
            random_tree_1.test_semantics),
            torch.mul(torch.sub(1, random_tree_1.test_semantics),
                      torch.add(tree2.test_semantics, torch.mul(ms, torch.sub(random_tree_4.test_semantics, random_tree_5.test_semantics)))
                      ))

    else:
        return torch.add(torch.mul(
            torch.add(tree1.train_semantics,
                      torch.mul(ms, torch.sub(random_tree_2.train_semantics, random_tree_3.train_semantics))),
            random_tree_1.train_semantics),
            torch.mul(torch.sub(1, random_tree_1.train_semantics),
                      torch.add(tree2.train_semantics,
                                torch.mul(ms, torch.sub(random_tree_4.train_semantics, random_tree_5.train_semantics)))
                      ))

