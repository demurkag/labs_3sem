#pragma once

#include "imodel.hpp"
#include "cart_classifier.hpp"
#include "random_forest_params.hpp"

#include <span>
#include <vector>

class RandomForestClassifier : public IClassifier
{
private:
    RandomForestParams params_;
    std::vector<CartClassifier> trees_;

public:
    RandomForestClassifier();

    explicit RandomForestClassifier(
        RandomForestParams params);

    void fit(
        const Matrix &X,
        std::span<const double> y) override;

    std::vector<double> predict(
        const Matrix &X) const override;

    Matrix predict_proba(
        const Matrix &X) const override;

    std::span<const double> classes() const override;
};