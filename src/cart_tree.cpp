#include "cart_tree.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

CartTree::CartTree(
    SplitCriterion criterion,
    std::size_t max_depth,
    std::size_t min_samples_split,
    std::size_t min_samples_leaf)
    : criterion_(criterion),
      max_depth_(max_depth),
      min_samples_split_(min_samples_split),
      min_samples_leaf_(min_samples_leaf)
{
    if (criterion_ != SplitCriterion::Variance &&
        criterion_ != SplitCriterion::Gini)
    {
        throw std::invalid_argument("Неизвестный критерий разбиения");
    }

    if (min_samples_split_ < 2)
    {
        throw std::invalid_argument(
            "min_samples_split должен быть не меньше 2");
    }

    if (min_samples_leaf_ == 0)
    {
        throw std::invalid_argument(
            "min_samples_leaf должен быть больше нуля");
    }
}

bool CartTree::is_fitted() const noexcept
{
    return root_ != nullptr;
}

std::span<const double> CartTree::classes() const
{
    if (!is_fitted())
    {
        throw std::logic_error("Дерево ещё не обучено");
    }

    return classes_;
}

std::unique_ptr<Node> CartTree::make_leaf(
    std::span<const double> y,
    const std::vector<std::size_t> &object_ids)
{
    auto node = std::make_unique<Node>();

    if (criterion_ == SplitCriterion::Variance)
    {
        double sum = 0.0;

        for (std::size_t i : object_ids)
        {
            sum += y[i];
        }

        node->value = sum / object_ids.size();
    }
    else if (criterion_ == SplitCriterion::Gini)
    {
        node->probabilities.assign(classes_.size(), 0.0);

        for (std::size_t i : object_ids)
        {
            auto it = std::lower_bound(
                classes_.begin(), classes_.end(), y[i]);

            std::size_t c = it - classes_.begin();

            node->probabilities[c] += 1.0;
        }

        for (double &probability : node->probabilities)
        {
            probability /= object_ids.size();
        }
    }

    return node;
}

std::unique_ptr<Node> CartTree::build_tree(
    const Matrix &X,
    std::span<const double> y,
    const std::vector<std::size_t> &object_ids,
    const std::vector<std::size_t> &feature_ids,
    std::size_t depth)
{
    if (depth >= max_depth_ ||
        object_ids.size() < min_samples_split_)
    {
        return make_leaf(y, object_ids);
    }

    bool all_equal = true;

    for (std::size_t i : object_ids)
    {
        if (y[i] != y[object_ids[0]])
        {
            all_equal = false;
            break;
        }
    }

    if (all_equal)
    {
        return make_leaf(y, object_ids);
    }

    auto split = finder_.find_best(
        X,
        y,
        object_ids,
        feature_ids,
        criterion_,
        min_samples_leaf_);

    if (!split.has_value())
    {
        return make_leaf(y, object_ids);
    }

    std::vector<std::size_t> left_ids;
    std::vector<std::size_t> right_ids;

    for (std::size_t i : object_ids)
    {
        if (X(i, split->feature) <= split->threshold)
        {
            left_ids.push_back(i);
        }
        else
        {
            right_ids.push_back(i);
        }
    }

    auto node = std::make_unique<Node>();

    node->is_leaf = false;
    node->feature = split->feature;
    node->threshold = split->threshold;

    node->left = build_tree(
        X, y, left_ids, feature_ids, depth + 1);

    node->right = build_tree(
        X, y, right_ids, feature_ids, depth + 1);

    return node;
}

void CartTree::validate_predict_input(const Matrix &X) const
{
    if (!is_fitted())
    {
        throw std::logic_error("Дерево ещё не обучено");
    }

    if (X.cols() != n_features_)
    {
        throw std::invalid_argument(
            "Количество признаков не совпадает с обучающей выборкой");
    }

    for (std::size_t i = 0; i < X.rows(); ++i)
    {
        for (std::size_t j = 0; j < X.cols(); ++j)
        {
            if (!std::isfinite(X(i, j)))
            {
                throw std::invalid_argument(
                    "Матрица содержит NaN или бесконечность");
            }
        }
    }
}

void CartTree::fit(const Matrix &X, std::span<const double> y)
{
    root_.reset();
    n_features_ = 0;
    classes_.clear();

    if (X.rows() == 0)
        throw std::invalid_argument("Обучающая выборка пуста");

    if (X.cols() == 0)
        throw std::invalid_argument("Матрица не содержит признаков");

    if (y.size() != X.rows())
        throw std::invalid_argument(
            "Количество строк матрицы X не совпадает с размером y");

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

    if (criterion_ == SplitCriterion::Gini)
    {
        classes_.assign(y.begin(), y.end());

        std::sort(classes_.begin(), classes_.end());

        classes_.erase(
            std::unique(classes_.begin(), classes_.end()),
            classes_.end());
    }

    std::vector<std::size_t> object_ids(X.rows());
    std::vector<std::size_t> feature_ids(X.cols());

    std::iota(object_ids.begin(), object_ids.end(), 0);
    std::iota(feature_ids.begin(), feature_ids.end(), 0);

    root_ = build_tree(X, y, object_ids, feature_ids, 0);

    n_features_ = X.cols();
}

const Node &CartTree::find_leaf(
    const Matrix &X,
    std::size_t row) const
{
    if (!is_fitted())
    {
        throw std::logic_error("Дерево ещё не обучено");
    }

    if (X.cols() != n_features_)
    {
        throw std::invalid_argument(
            "Количество признаков не совпадает с обучающей выборкой");
    }

    if (row >= X.rows())
    {
        throw std::out_of_range("Некорректный индекс строки");
    }

    const Node *current = root_.get();

    while (!current->is_leaf)
    {
        if (X(row, current->feature) <= current->threshold)
        {
            current = current->left.get();
        }
        else
        {
            current = current->right.get();
        }
    }

    return *current;
}