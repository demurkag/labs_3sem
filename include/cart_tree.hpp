#pragma once

#include "cart_node.hpp"
#include "split_finder.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

class CartTree
{
private:
    std::unique_ptr<Node> root_;

    SplitCriterion criterion_;

    std::size_t max_depth_;
    std::size_t min_samples_split_;
    std::size_t min_samples_leaf_;

    std::size_t n_features_ = 0;

    SplitFinder finder_;

    std::vector<double> classes_;

    std::unique_ptr<Node> build_tree(
        const Matrix &X,
        std::span<const double> y,
        const std::vector<std::size_t> &object_ids,
        const std::vector<std::size_t> &feature_ids,
        std::size_t depth);

    std::unique_ptr<Node> make_leaf(
        std::span<const double> y,
        const std::vector<std::size_t> &object_ids);

public:
    CartTree(
        SplitCriterion criterion,
        std::size_t max_depth,
        std::size_t min_samples_split,
        std::size_t min_samples_leaf);

    void validate_predict_input(const Matrix &X) const;

    void fit(const Matrix &X, std::span<const double> y);

    const Node &find_leaf(
        const Matrix &X,
        std::size_t row) const;

    bool is_fitted() const noexcept;

    std::span<const double> classes() const;
};
