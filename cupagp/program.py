from __future__ import annotations

from typing import Tuple
import random

from .fset import *

# =================================================================================
# Multi-Layer GP tree structure for binary image classification.
# =================================================================================
# Classification Layer =======> class_0 <= [result <= 0] | [result > 0] => class_1
#                                                        |
#                                                      [Sub]
#                                                     /     \
# Feature Construction Layer =>                 [G_Std]      [G_Std]
#                                                  |            |
#                                             [Hist_Eq]      [Lap]
#                                                  |            |
# Feature Extraction Layer ===>               [Sobel_X]      [Sobel_Y]
#                                                  |            |
# Region Detection Layer =====>              [Region_S]      [Region_R]
# =================================================================================


feature_construct_set = [G_Std, Sub]
terminal_set = [Region_S, Region_R]
inter_func_set = [Hist_Eq, Gau1, Lap, Sobel_X, Sobel_Y, LoG1, LoG2, LBP, Gau11, GauXY]


class _Node:
    def __init__(self, name=0, rx=0, ry=0, rh=0, rw=0):
        self.name = name
        self.rx = rx
        self.ry = ry
        self.rh = rh
        self.rw = rw

    def __repr__(self):
        return func_to_str(self.name)

    def __str__(self):
        return func_to_str(self.name) + '(' + str(self.rx) + ', ' + str(self.ry) + ', ' \
               + str(self.rh) + ', ' + str(self.rw) + ')' if self.is_terminal_node() else func_to_str(self.name)

    def is_binary_function_node(self) -> bool:
        return self.name == Sub

    def is_unary_function_node(self) -> bool:
        return self.name != Sub

    def is_terminal_node(self) -> bool:
        return self.name == Region_R or self.name == Region_S

    def rand_terminal_node(self, img_h, img_w):
        """Create a random terminal node with respect to the height and width of the image."""
        if img_h < 20 or img_w < 20:
            raise RuntimeError('The height and width must be larger or equal to 20.')

        reg_x = random.randint(0, img_h - 20)
        reg_y = random.randint(0, img_w - 20)

        if random.sample(population=terminal_set, k=1)[0] == Region_S:
            max_side_len = min(img_h - reg_x, img_w - reg_y)
            side_len = random.randint(20, max_side_len)
            self.name = Region_S
            self.rx = reg_x
            self.ry = reg_y
            self.rh = side_len
            self.rw = side_len
        else:
            reg_h = random.randint(20, img_h - reg_x)
            reg_w = random.randint(20, img_w - reg_y)
            self.name = Region_R
            self.rx = reg_x
            self.ry = reg_y
            self.rh = reg_h
            self.rw = reg_w

    def rand_inter_func_node(self):
        name = random.sample(population=inter_func_set, k=1)[0]
        self.name = name

    def rand_feature_construct_node(self):
        name = random.sample(population=feature_construct_set, k=1)[0]
        self.name = name


class _TreeGenerator:
    class TreeNode:
        def __init__(self, node=None, left=None, right=None):
            self.node = node
            self.left = left
            self.right = right

    def __init__(self, depth, img_h, img_w):
        self.depth = depth
        self.img_h = img_h
        self.img_w = img_w

    def full_init_tree(self) -> List[_Node]:
        root_tree_node = self._create_full_tree(self.depth)
        ret = []
        self._get_init_prefix(prefix=ret, tree_node=root_tree_node)
        return ret

    def growth_init_tree(self) -> List[_Node]:
        root_tree_node = self._create_growth_tree(self.depth)
        ret = []
        self._get_init_prefix(prefix=ret, tree_node=root_tree_node)
        return ret

    def _get_init_prefix(self, prefix: List[_Node], tree_node):
        if tree_node is None:
            return
        prefix.append(tree_node.node)
        self._get_init_prefix(prefix, tree_node.left)
        self._get_init_prefix(prefix, tree_node.right)

    def _create_full_tree(self, depth, parent_node=None) -> TreeNode:
        if parent_node is None:
            tree_node = self.TreeNode(node=_Node(Sub))
            tree_node.left = self._create_full_tree(depth - 1, Sub)
            tree_node.right = self._create_full_tree(depth - 1, Sub)
            return tree_node

        if depth == 2 and parent_node == Sub:
            node = _Node(name=G_Std)
            tree_node = self.TreeNode(node=node)
            tree_node.left = self._create_full_tree(depth - 1, G_Std)
            return tree_node

        if depth == 1:
            terminal = _Node()
            terminal.rand_terminal_node(self.img_h, self.img_w)
            tree_node = self.TreeNode(node=terminal)
            return tree_node

        if parent_node == Sub:
            node = _Node()
            node.rand_feature_construct_node()
            tree_node = self.TreeNode(node=node)
            if node.name == G_Std:
                tree_node.left = self._create_full_tree(depth - 1, G_Std)
                return tree_node
            else:
                tree_node.left = self._create_full_tree(depth - 1, Sub)
                tree_node.right = self._create_full_tree(depth - 1, Sub)
                return tree_node
        else:
            node = _Node()
            node.rand_inter_func_node()
            tree_node = self.TreeNode(node=node)
            tree_node.left = self._create_full_tree(depth - 1, node.name)
            return tree_node

    def _create_growth_tree(self, depth, parent_node=None, return_rate=0.5):
        if parent_node is None:
            tree_node = self.TreeNode(node=_Node(Sub))
            tree_node.left = self._create_growth_tree(depth - 1, Sub)
            tree_node.right = self._create_growth_tree(depth - 1, Sub)
            return tree_node

        if depth == 2 and parent_node == Sub:
            node = _Node(name=G_Std)  # must be S_Std, can not be Sub
            tree_node = self.TreeNode(node=node)
            tree_node.left = self._create_growth_tree(depth - 1, G_Std)
            return tree_node

        if depth == 1:
            terminal = _Node()
            terminal.rand_terminal_node(self.img_h, self.img_w)
            tree_node = self.TreeNode(node=terminal)
            return tree_node

        if parent_node is Sub:
            node = _Node()
            node.rand_feature_construct_node()
            tree_node = self.TreeNode(node=node)
            if node.name == G_Std:
                tree_node.left = self._create_growth_tree(depth - 1, G_Std)
                return tree_node

            else:
                tree_node.left = self._create_growth_tree(depth - 1, Sub)
                tree_node.right = self._create_growth_tree(depth - 1, Sub)
                return tree_node

        else:
            if random.random() < return_rate:
                node = _Node()
                node.rand_terminal_node(self.img_h, self.img_w)
            else:
                node = _Node()
                node.rand_inter_func_node()
            tree_node = self.TreeNode(node=node)

            if node.is_terminal_node():
                return tree_node

            tree_node.left = self._create_growth_tree(depth - 1, node.name)
            return tree_node


def _subtree_index(prefix: List[_Node], start_pos) -> Tuple[int, int]:
    func_count = 0
    term_count = 0
    end = start_pos
    while end < len(prefix):
        node = prefix[end]
        if node.is_binary_function_node():
            func_count += 1
        elif node.is_terminal_node():
            term_count += 1
        if func_count + 1 == term_count:
            break
        end += 1
    return start_pos, end + 1


class Program:
    def __init__(self, img_h, img_w, init_depth=6, init_method=None, prefix=None):
        """Program for GP based image binary classification.
        Args:
            img_h      : image height
            img_w      : image width
            init_method: 'full' or 'growth'
            prefix     : init by given prefix
        """
        self.img_h = img_h
        self.img_w = img_w
        self.prefix: List[_Node] = []
        self.depth = 0
        self.fitness = 0
        self._init_depth = init_depth
        self._init_method = init_method

        # init by prefix
        if prefix is not None:
            self.prefix = prefix
            self.get_depth_of_program()
            return

        # init respect  to the init method
        if init_method is not None:
            if init_method == 'full':
                gener = _TreeGenerator(depth=init_depth, img_h=img_h, img_w=img_w)
                self.prefix = gener.full_init_tree()
                self.get_depth_of_program()
            elif init_method == 'growth':
                gener = _TreeGenerator(depth=init_depth, img_h=img_h, img_w=img_w)
                self.prefix = gener.growth_init_tree()
                self.get_depth_of_program()
            else:
                raise RuntimeError("init_method must be 'full' or 'growth'.")

    def __repr__(self):
        ret = '[ '
        for node in self.prefix:
            ret += str(node) + ' '
        ret += ']'
        return ret

    def __len__(self):
        return len(self.prefix)

    def __getitem__(self, item):
        return self.prefix[item]

    def get_depth_of_program(self):
        s = []
        for node in reversed(self.prefix):
            if node.is_terminal_node():
                s.append(1)
            elif node.is_binary_function_node():
                depth0 = s.pop()
                depth1 = s.pop()
                s.append(max(depth0, depth1) + 1)
            else:
                depth = s.pop()
                s.append(depth + 1)
        self.depth = s.pop()

    def crossover(self, donor: Program | List[_Node]):
        if isinstance(donor, Program):
            dprefix = donor.prefix
        else:
            dprefix = donor
        self_start = random.randint(1, len(self.prefix) - 1)
        self_start, self_end = _subtree_index(prefix=self.prefix, start_pos=self_start)

        root_func = self.prefix[self_start].name
        if root_func == Sub:
            root_indexes = [i for i in range(len(donor)) if donor[i].name == Sub or donor[i].name == G_Std]
            rand_start_pos = random.sample(population=root_indexes, k=1)[0]
            donor_start, donor_end = _subtree_index(prefix=dprefix, start_pos=rand_start_pos)

        elif root_func == G_Std:
            root_indexes = [i for i in range(len(donor)) if donor[i].name == G_Std]
            rand_start_pos = random.sample(population=root_indexes, k=1)[0]
            donor_start, donor_end = _subtree_index(prefix=dprefix, start_pos=rand_start_pos)

        else:
            root_indexes = [i for i in range(len(donor)) if donor[i].name != Sub and donor[i].name != G_Std]
            rand_start_pos = random.sample(population=root_indexes, k=1)[0]
            donor_start, donor_end = _subtree_index(prefix=dprefix, start_pos=rand_start_pos)

        # gen new prefix
        ret_prefix = [self[i] for i in range(0, self_start)]
        ret_prefix += [donor[i] for i in range(donor_start, donor_end)]
        ret_prefix += [self[i] for i in range(self_end, len(self))]

        self.prefix = ret_prefix
        self.get_depth_of_program()

    def point_mutation(self):
        point_indexes = [i for i in range(len(self.prefix))
                         if self.prefix[i].name != Sub and self.prefix[i].name != G_Std]
        pos = random.sample(population=point_indexes, k=1)[0]

        if self[pos].is_terminal_node():
            node = _Node()
            node.rand_terminal_node(self.img_h, self.img_w)
            self.prefix[pos] = node
        else:
            node = _Node()
            node.rand_inter_func_node()
            self.prefix[pos] = node

    def subtree_mutation(self):
        gener = _TreeGenerator(depth=self._init_depth, img_h=self.img_h, img_w=self.img_w)
        if self._init_method == 'full':
            rand_tree = gener.full_init_tree()
        elif self._init_method == 'growth':
            rand_tree = gener.growth_init_tree()
        else:
            if random.random() < 0.5:
                rand_tree = gener.full_init_tree()
            else:
                rand_tree = gener.growth_init_tree()
        return self.crossover(rand_tree)

    def hoist_mutation(self):
        # random subtree of self.prefix
        root_indexes = [i for i in range(1, len(self.prefix)) if self.prefix[i].name != G_Std]
        rand_start_pos = random.sample(population=root_indexes, k=1)[0]
        start1, end1 = _subtree_index(prefix=self.prefix, start_pos=rand_start_pos)
        subtree = [self.prefix[i] for i in range(start1, end1)]

        if self.prefix[start1].name == Sub:
            root_indexes = [i for i in range(len(subtree))
                            if subtree[i].name == Sub or subtree[i].name == G_Std]
            rand_start_pos = random.sample(population=root_indexes, k=1)[0]
            start2, end2 = _subtree_index(prefix=subtree, start_pos=rand_start_pos)
        else:
            start2 = random.randint(0, len(subtree) - 1)
            start2, end2 = _subtree_index(prefix=subtree, start_pos=start2)

        # gen new prefix
        ret = [self.prefix[i] for i in range(0, start1)]
        ret += [subtree[i] for i in range(start2, end2)]
        ret += [self.prefix[i] for i in range(end1, len(self))]

        self.prefix = ret
        self.get_depth_of_program()
