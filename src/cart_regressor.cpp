#include "cart_regressor.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

CartRegressor::CartRegressor(
    std::size_t max_depth,
    std::size_t min_samples_split,
    std::size_t min_samples_leaf)
    : tree_(
          SplitCriterion::Variance,
          max_depth,
          min_samples_split,
          min_samples_leaf)
{
}

void CartRegressor::fit(
    const Matrix &X,
    std::span<const double> y)
{
    tree_.fit(X, y);
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
