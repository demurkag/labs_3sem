#pragma once

#include "imodel.hpp"
#include "cart_tree.hpp"

#include <cstddef>
#include <span>
#include <vector>

class CartRegressor : public IRegressor
{
private:
    CartTree tree_;

public:
    CartRegressor(
        std::size_t max_depth,
        std::size_t min_samples_split,
        std::size_t min_samples_leaf);

    void fit(
        const Matrix &X,
        std::span<const double> y) override;

    std::vector<double> predict(
        const Matrix &X) const override;
};
