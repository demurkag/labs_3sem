#include "cart_classifier.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

CartClassifier::CartClassifier(
    std::size_t max_depth,
    std::size_t min_samples_split,
    std::size_t min_samples_leaf)
    : tree_(
          SplitCriterion::Gini,
          max_depth,
          min_samples_split,
          min_samples_leaf)
{
}

void CartClassifier::fit(
    const Matrix &X,
    std::span<const double> y)
{
    tree_.fit(X, y);
}

std::vector<double> CartClassifier::predict(
    const Matrix &X) const
{
    tree_.validate_predict_input(X);

    const auto labels = tree_.classes();
    std::vector<double> predictions(X.rows());

    for (std::size_t i = 0; i < X.rows(); ++i)
    {
        const Node &leaf = tree_.find_leaf(X, i);

        const auto it = std::max_element(
            leaf.probabilities.begin(),
            leaf.probabilities.end());

        const std::size_t class_index =
            static_cast<std::size_t>(it - leaf.probabilities.begin());

        predictions[i] = labels[class_index];
    }

    return predictions;
}

Matrix CartClassifier::predict_proba(
    const Matrix &X) const
{
    tree_.validate_predict_input(X);

    const auto labels = tree_.classes();
    Matrix result(X.rows(), labels.size(), Layout::RowMajor);

    for (std::size_t i = 0; i < X.rows(); ++i)
    {
        const Node &leaf = tree_.find_leaf(X, i);

        for (std::size_t c = 0; c < labels.size(); ++c)
        {
            result(i, c) = leaf.probabilities[c];
        }
    }

    return result;
}

std::span<const double> CartClassifier::classes() const
{
    return tree_.classes();
}

TreeStats CartClassifier::stats() const
{
    return tree_.stats();
}

std::vector<SplitInfo> CartClassifier::first_splits(
    std::size_t limit) const
{
    return tree_.first_splits(limit);
}