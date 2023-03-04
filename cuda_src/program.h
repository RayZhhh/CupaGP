//
// Created by Derek on 2022/11/14.
//

#ifndef CUGPIBC_PROGRAM_H
#define CUGPIBC_PROGRAM_H

// =================================================================================
// Multi-Layer GP tree structure for binary image classification.
// =================================================================================
//
// Classification Layer =======> class_0 <= [result <= 0] | [result > 0] => class_1
//                                                        |
//                                                      [Sub]
//                                                     /     \
// Feature Construction Layer =>                 [G_Std]      [G_Std]
//                                                  |            |
//                                             [Hist_Eq]      [Lap]
//                                                  |            |
// Feature Extraction Layer ===>               [Sobel_X]      [Sobel_Y]
//                                                  |            |
// Region Detection Layer =====>              [Region_S]      [Region_R]
// =================================================================================

#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <cassert>
#include "rand_engine.h"
#include <stack>

using namespace std;


typedef vector<int> vecI;
//using vecI = vector<int>;


enum F {
    G_Std, Sub, Region_S, Region_R, Hist_Eq, Gau1, Lap, Sobel_X, Sobel_Y, LoG, LoG1, LoG2, LBP, Gau11, GauXY
};


string func_to_str(int f) {
    string strs[] = {"G_Std", "Sub", "Region_S", "Region_R", "Hist_Eq", "Gau1", "Lap",
                     "Sobel_X", "Sobel_Y", "LoG", "LoG1", "LoG2", "LBP", "Gau11", "GauXY"};
    return strs[f];
}


static vecI feature_construct_set = {G_Std, Sub}; // NOLINT
static vecI terminal_set = {Region_R, Region_S}; // NOLINT
static vecI inter_func_set = {Hist_Eq, Gau1, Lap, Sobel_X, Sobel_Y, LoG, LoG1, LoG2, LBP, Gau11, GauXY}; // NOLINT


class Node {
public:

    Node(int n, int x, int y, int h, int w) : name(n), rx(x), ry(y), rh(h), rw(w) {}

    explicit Node(int n) : name(n), rx(0), ry(0), rh(0), rw(0) {}

    Node() = default;

    int name{};
    int rx{};
    int ry{};
    int rh{};
    int rw{};

    bool is_terminal() const {
        if (name == Region_S || name == Region_R)
            return true;
        return false;
    }

    bool is_binary_func() const {
        return name == Sub;
    }

    string str() const {
        if (is_terminal()) {
            return func_to_str(name) + "(" + to_string(rx) + ", " + to_string(ry) + ", " + to_string(rh) +
                   ", " + to_string(rw) + ")";
        } else {
            return func_to_str(name);
        }
    }

    void rand_term(int h, int w) {
        assert(h > 20 && w > 20);

        rx = randint_(0, h - 20);
        ry = randint_(0, w - 20);

        int term = terminal_set[randint_(0, terminal_set.size() - 1)];
        name = term;

        if (term == Region_S) {
            auto max_side_len = min(h - rx, w - ry);
            auto side_len = randint_(20, max_side_len);
            rh = side_len;
            rw = side_len;
        } else {
            rh = randint_(20, h - rx);
            rw = randint_(20, w - ry);
        }
    }

    void rand_inter_func() {
        int func = inter_func_set[randint_(0, inter_func_set.size() - 1)];
        name = func;
    }

    void rand_feature_construct() {
        int func = feature_construct_set[randint_(0, feature_construct_set.size() - 1)];
        name = func;
    }
};


typedef vector<Node> Prefix;


class Program {
public:
    Program() = default;

    Program(int _img_h, int _img_w, int _init_depth, string _init_method)
            : h(_img_h), w(_img_w), init_depth(_init_depth), init_method(std::move(_init_method)) {
        if (init_method == "full") {
            full_init(init_depth);
        } else {
            grow_init(init_depth);
        }
    }

    Prefix inner_prefix;
    int depth{};
    float fitness{};

    int h;
    int w;
    int init_depth{};
    string init_method;

    void get_depth_of_program() {
        stack<int> s;
        for (int i = inner_prefix.size() - 1; i >= 0; i--) {
            if (inner_prefix[i].is_terminal()) {
                s.push(1);
            } else if (inner_prefix[i].is_binary_func()) {
                int depth0 = s.top();
                s.pop();
                int depth1 = s.top();
                s.pop();
                s.push(max(depth0, depth1) + 1);
            } else {
                int depth_ = s.top();
                s.pop();
                s.push(depth_ + 1);
            }
        }
        assert(s.size() == 1);
        this->depth = s.top();
    }

    void crossover(Program &donor) {
        this->prefix_crossover(donor.inner_prefix);
        get_depth_of_program();
    }

    void subtree_mutation(float _full_rate = 0.5) {
        if (random_() < _full_rate) {
            Program donor(h, w, init_depth, "full");
            this->prefix_crossover(donor.inner_prefix);
            get_depth_of_program();
        } else {
            Program donor(h, w, init_depth, "grow");
            this->prefix_crossover(donor.inner_prefix);
            get_depth_of_program();
        }
    }

    void point_mutation() {
        vecI point_indexes;
        for (int i = 0; i < inner_prefix.size(); i++) {
            if (inner_prefix[i].name != Sub && inner_prefix[i].name != G_Std) {
                point_indexes.emplace_back(i);
            }
        }
        int pos = point_indexes[randint_(0, point_indexes.size() - 1)];
        if (inner_prefix[pos].is_terminal()) {
            inner_prefix[pos].rand_term(h, w);
        } else {
            inner_prefix[pos].rand_inter_func();
        }
    }

    void hoist_mutation() {
        vecI root_indexes;
        for (int i = 1; i < inner_prefix.size(); i++) {
            if (inner_prefix[i].name != G_Std) {
                root_indexes.emplace_back(i);
            }
        }

        int start1 = root_indexes[randint_(0, root_indexes.size() - 1)];
        int end1 = subtree_index(inner_prefix, start1);
        int start2, end2;

        // random_ subtree of inner prefix
        Prefix subtree;
        for (int i = start1; i < end1; i++) subtree.emplace_back(inner_prefix[i]);

        if (inner_prefix[start1].name == Sub) {
            root_indexes.clear();
            for (int i = 0; i < subtree.size(); i++)
                if (subtree[i].name == Sub || subtree[i].name == G_Std)
                    root_indexes.emplace_back(i);

            start2 = root_indexes[randint_(0, root_indexes.size() - 1)];
            end2 = subtree_index(subtree, start2);
        } else {
            start2 = randint_(0, subtree.size() - 1);
            end2 = subtree_index(subtree, start2);
        }

        Prefix ret;
        for (int i = 0; i < start1; i++) ret.emplace_back(inner_prefix[i]);
        for (int i = start2; i < end2; i++) ret.emplace_back(subtree[i]);
        for (int i = end1; i < inner_prefix.size(); i++) ret.emplace_back(inner_prefix[i]);

        inner_prefix = ret;
        get_depth_of_program();
    }

    string str() const {
        string ret = "[ ";
        for (auto n: inner_prefix) {
            ret += n.str();
            ret += " ";
        }
        return ret + "]";
    }

protected:
    void full_init(int _init_depth) {
        inner_prefix.clear();
        auto tree_node = create_full_tree(_init_depth);
        get_prefix(tree_node);
        delete tree_node;
        get_depth_of_program();
    }

    void grow_init(int _init_depth) {
        inner_prefix.clear();
        auto tree_node = create_grow_tree(_init_depth);
        get_prefix(tree_node);
        delete tree_node;
        get_depth_of_program();
    }

    void prefix_crossover(Prefix &donor) {
        int self_start = randint_(1, inner_prefix.size() - 1);
        int self_end = subtree_index(inner_prefix, self_start);

        int donor_start;
        int donor_end;

        auto root_func = inner_prefix[self_start].name;
        if (root_func == Sub) {
            vecI root_indexes;
            for (int i = 0; i < donor.size(); i++) {
                if (donor[i].name == Sub || donor[i].name == G_Std) {
                    root_indexes.emplace_back(i);
                }
            }
            donor_start = root_indexes[randint_(0, root_indexes.size() - 1)];
            donor_end = subtree_index(donor, donor_start);
        } else if (root_func == G_Std) {
            vecI root_indexes;
            for (int i = 0; i < donor.size(); i++) {
                if (donor[i].name == G_Std) {
                    root_indexes.emplace_back(i);
                }
            }
            donor_start = root_indexes[randint_(0, root_indexes.size() - 1)];
            donor_end = subtree_index(donor, donor_start);
        } else { // any inter func
            vecI root_indexes;
            for (int i = 0; i < donor.size(); i++) {
                if (donor[i].name != G_Std && donor[i].name != Sub) {
                    root_indexes.emplace_back(i);
                }
            }
            donor_start = root_indexes[randint_(0, root_indexes.size() - 1)];
            donor_end = subtree_index(donor, donor_start);
        }

        // gen new prefix
        Prefix ret_prefix;
        for (int i = 0; i < self_start; i++) ret_prefix.emplace_back(inner_prefix[i]);
        for (int i = donor_start; i < donor_end; i++) ret_prefix.emplace_back(donor[i]);
        for (int i = self_end; i < inner_prefix.size(); i++) ret_prefix.emplace_back(inner_prefix[i]);
        inner_prefix = ret_prefix;
    }

    static int subtree_index(const Prefix &_prefix, int start_pos) {
        int func_count = 0;
        int term_count = 0;
        int end = start_pos;

        while (end < _prefix.size()) {
            auto &node = _prefix[end];
            if (node.is_binary_func()) {
                func_count++;
            } else if (node.is_terminal()) {
                term_count++;
            }

            if (func_count + 1 == term_count) {
                break;
            }

            end++;
        }
        return end + 1;
    }

    class TreeNode {
    public:
        TreeNode(Node n) {
            node = n;
        }

        Node node;
        TreeNode *left = nullptr;
        TreeNode *right = nullptr;
    };

    void get_prefix(TreeNode *tree_node) {
        if (tree_node == nullptr) {
            return;
        }
        inner_prefix.emplace_back(tree_node->node);
        get_prefix(tree_node->left);
        delete tree_node->left;
        get_prefix(tree_node->right);
        delete tree_node->right;
    }

    TreeNode *create_full_tree(int cur_depth, int parent = -1) {
        if (parent == -1) {
            auto tree_node = new TreeNode(Node(Sub));
            tree_node->left = create_full_tree(cur_depth - 1, Sub);
            tree_node->right = create_full_tree(cur_depth - 1, Sub);
            return tree_node;
        }

        if (cur_depth == 2 && parent == Sub) {
            auto tree_node = new TreeNode(Node(G_Std));
            tree_node->left = create_full_tree(cur_depth - 1, G_Std);
            return tree_node;
        }

        if (cur_depth == 1) {
            auto term = Node();
            term.rand_term(h, w);

            auto tree_node = new TreeNode(term);
            return tree_node;
        }

        if (parent == Sub) {
            auto node = Node();
            node.rand_feature_construct();
            auto tree_node = new TreeNode(node);
            if (node.name == G_Std) {
                tree_node->left = create_full_tree(cur_depth - 1, G_Std);
                return tree_node;
            } else {
                tree_node->left = create_full_tree(cur_depth - 1, Sub);
                tree_node->right = create_full_tree(cur_depth - 1, Sub);
                return tree_node;
            }
        } else {
            auto node = Node();
            node.rand_inter_func();
            auto tree_node = new TreeNode(node);
            tree_node->left = create_full_tree(cur_depth - 1, node.name);
            return tree_node;
        }
    }

    TreeNode *create_grow_tree(int cur_depth, int parent = -1, float return_rate = 0.5) {
        if (parent == -1) {
            auto tree_node = new TreeNode(Node(Sub));
            tree_node->left = create_grow_tree(cur_depth - 1, Sub);
            tree_node->right = create_grow_tree(cur_depth - 1, Sub);
            return tree_node;
        }

        if (cur_depth == 2 && parent == Sub) {
            auto tree_node = new TreeNode(Node(G_Std));
            tree_node->left = create_grow_tree(cur_depth - 1, G_Std);
            return tree_node;
        }

        if (cur_depth == 1) {
            auto term = Node();
            term.rand_term(h, w);
            auto tree_node = new TreeNode(term);
            return tree_node;
        }

        if (parent == Sub) {
            auto node = Node();
            node.rand_feature_construct();
            auto tree_node = new TreeNode(node);
            if (node.name == G_Std) {
                tree_node->left = create_grow_tree(cur_depth - 1, G_Std);
                return tree_node;
            } else {
                tree_node->left = create_grow_tree(cur_depth - 1, Sub);
                tree_node->right = create_grow_tree(cur_depth - 1, Sub);
                return tree_node;
            }
        } else {
            auto node = Node();
            if (random_() < return_rate) {
                node.rand_term(h, w);
            } else {
                node.rand_inter_func();
            }
            auto tree_node = new TreeNode(node);

            if (node.is_terminal()) {
                return tree_node;
            }

            tree_node->left = create_grow_tree(cur_depth - 1, node.name);
            return tree_node;
        }
    }
};

#endif //CUGPIBC_PROGRAM_H
