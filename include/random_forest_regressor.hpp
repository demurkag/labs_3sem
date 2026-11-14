#pragma once

#include "imodel.hpp"
#include "cart_regressor.hpp"
#include "random_forest_params.hpp"

class RandomForestRegressor : public IRegressor
{
private:
    RandomForestParams params_;
    std::vector<CartRegressor> trees_;

public:
    explicit RandomForestRegressor(
        RandomForestParams params = {});

    void fit(
        const Matrix &X,
        std::span<const double> y) override;

    std::vector<double> predict(
        const Matrix &X) const override;
};