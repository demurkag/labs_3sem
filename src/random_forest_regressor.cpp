#include "random_forest_regressor.hpp"

#include <stdexcept>
#include <cmath>
#include <random>
#include <algorithm>
#include <utility>

RandomForestRegressor::RandomForestRegressor(
    RandomForestParams params)
    : params_(params)
{
    if (params_.n_estimators == 0)
        throw std::invalid_argument(
            "Число деревьев должно быть не меньше 1");

    if (params_.min_samples_split < 2)
        throw std::invalid_argument(
            "min_samples_split должен быть не меньше 2");

    if (params_.min_samples_leaf == 0)
        throw std::invalid_argument(
            "min_samples_leaf должен быть больше нуля");

    if (params_.n_jobs == 0)
        throw std::invalid_argument(
            "n_jobs должен быть больше нуля");

    if (params_.max_depth.has_value() &&
        *params_.max_depth == 0)
        throw std::invalid_argument(
            "max_depth должен быть больше нуля");

    if (params_.max_samples.has_value() &&
        *params_.max_samples == 0)
        throw std::invalid_argument(
            "max_samples должен быть больше нуля");

    if (!params_.bootstrap &&
        params_.max_samples.has_value())
        throw std::invalid_argument(
            "max_samples нельзя задавать при bootstrap=false");

    if (params_.max_features.mode != FeatureMode::All &&
        params_.max_features.mode != FeatureMode::Sqrt &&
        params_.max_features.mode != FeatureMode::Count)
        throw std::invalid_argument(
            "Неизвестный режим max_features");

    if (params_.max_features.mode == FeatureMode::Count &&
        params_.max_features.count == 0)
        throw std::invalid_argument(
            "max_features.count должен быть больше нуля в режиме Count");
}

void RandomForestRegressor::fit(
    const Matrix &X,
    std::span<const double> y)
{
    trees_.clear();

    if (X.rows() == 0 || X.cols() == 0)
        throw std::invalid_argument(
            "Обучающая матрица не должна быть пустой");

    if (y.size() != X.rows())
        throw std::invalid_argument(
            "Размер y должен совпадать с числом строк X");

    if (params_.max_samples.has_value() &&
        *params_.max_samples > X.rows())
        throw std::invalid_argument(
            "max_samples не должен превышать число обучающих объектов");

    if (params_.max_features.mode == FeatureMode::Count &&
        params_.max_features.count > X.cols())
        throw std::invalid_argument(
            "max_features.count не должен превышать число признаков");

    for (double value : y)
    {
        if (!std::isfinite(value))
            throw std::invalid_argument("Отклик содержит NaN или бесконечность");
    }

    for (std::size_t i = 0; i < X.rows(); ++i)
    {
        for (std::size_t j = 0; j < X.cols(); ++j)
        {
            if (!std::isfinite(X(i, j)))
                throw std::invalid_argument(
                    "Матрица содержит NaN или бесконечность");
        }
    }

    const std::size_t n = X.rows();

    std::size_t m;

    if (params_.max_samples.has_value())
    {
        m = *params_.max_samples;
    }
    else
    {
        m = n;
    }

    std::mt19937 rng(params_.random_state);

    std::vector<std::uint32_t> seeds;
    seeds.reserve(params_.n_estimators);

    while (seeds.size() < params_.n_estimators)
    {
        std::uint32_t seed = rng();
        if (std::find(seeds.begin(), seeds.end(), seed) == seeds.end())
            seeds.push_back(seed);
    }

    std::vector<CartRegressor> new_trees;
    new_trees.reserve(params_.n_estimators);

    for (std::size_t i = 0; i < params_.n_estimators; ++i)
    {
        std::mt19937 tree_rng(seeds[i]);
        std::vector<std::size_t> counts(n, 0);

        if (!params_.bootstrap)
        {
            std::fill(counts.begin(), counts.end(), 1);
        }
        else
        {
            std::uniform_int_distribution<std::size_t> dist(0, n - 1);

            for (std::size_t j = 0; j < m; ++j)
            {
                std::size_t index = dist(tree_rng);
                counts[index]++;
            }
        }

        CartRegressor tree(
            params_.max_depth,
            params_.min_samples_split,
            params_.min_samples_leaf,
            params_.max_features,
            seeds[i]);

        tree.fit_with_counts(X, y, counts);

        new_trees.push_back(std::move(tree));
    }
    trees_ = std::move(new_trees);
}

std::vector<double> RandomForestRegressor::predict(
    const Matrix &X) const
{
    if (trees_.empty())
        throw std::logic_error("Случайный лес не обучен");

    std::vector<double> result(X.rows(), 0.0);

    for (const CartRegressor &tree : trees_)
    {
        std::vector<double> predictions = tree.predict(X);

        for (std::size_t j = 0; j < X.rows(); ++j)
        {
            result[j] += predictions[j];
        }
    }

    for (double &value : result)
    {
        value /= trees_.size();
    }
    return result;
}