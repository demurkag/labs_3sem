#include "cart_regressor.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>
#include <cstdint>

CartRegressor::CartRegressor(
    std::optional<std::size_t> max_depth,
    std::size_t min_samples_split,
    std::size_t min_samples_leaf,
    FeatureSearchParams feature_search,
    std::uint32_t random_state)
    : tree_(
          SplitCriterion::Variance,
          max_depth,
          min_samples_split,
          min_samples_leaf,
          feature_search,
          random_state)
{
}

void CartRegressor::fit(
    const Matrix &X,
    std::span<const double> y)
{
    tree_.fit(X, y);
}

void CartRegressor::fit_with_counts(
    const Matrix &X,
    std::span<const double> y,
    std::span<const std::size_t> counts)
{
    tree_.fit_with_counts(X, y, counts);
}

std::vector<double> CartRegressor::predict(
    const Matrix &X) const
{
    tree_.validate_predict_input(X);

    std::vector<double> predictions(X.rows());

    for (std::size_t i = 0; i < X.rows(); ++i)
    {
        const Node &leaf = tree_.find_leaf(X, i);
        predictions[i] = leaf.value;
    }
    return predictions;
}

TreeStats CartRegressor::stats() const
{
    return tree_.stats();
}

std::vector<SplitInfo> CartRegressor::first_splits(
    std::size_t limit) const
{
    return tree_.first_splits(limit);
}