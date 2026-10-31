#pragma once

#include <cstddef>
#include <memory>
#include <vector>

struct Node
{
    bool is_leaf = true;

    std::size_t feature = 0;
    double threshold = 0.0;

    double value = 0.0;
    std::vector<double> probabilities;

    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
};
