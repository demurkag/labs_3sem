#include "cart_tree.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <algorithm>

CartTree::CartTree(
    SplitCriterion criterion,
    std::optional<std::size_t> max_depth,
    std::size_t min_samples_split,
    std::size_t min_samples_leaf,
    FeatureSearchParams feature_search,
    std::uint32_t random_state)
    : criterion_(criterion),
      max_depth_(max_depth),
      min_samples_split_(min_samples_split),
      min_samples_leaf_(min_samples_leaf),
      feature_search_(feature_search),
      random_state_(random_state),
      rng_(random_state)
{
    if (criterion_ != SplitCriterion::Variance &&
        criterion_ != SplitCriterion::Gini)
    {
        throw std::invalid_argument("Неизвестный критерий разбиения");
    }

    if (feature_search_.mode != FeatureMode::All &&
        feature_search_.mode != FeatureMode::Sqrt &&
        feature_search_.mode != FeatureMode::Count)
    {
        throw std::invalid_argument(
            "Неизвестный режим выбора признаков");
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

    if (feature_search_.mode == FeatureMode::Count &&
        feature_search_.count == 0)
    {
        throw std::invalid_argument(
            "Количество признаков должно быть больше нуля");
    }
}

std::size_t CartTree::resolve_max_features(std::size_t d) const
{
    if (feature_search_.mode == FeatureMode::All)
        return d;

    if (feature_search_.mode == FeatureMode::Sqrt)
        return std::max(
            std::size_t{1},
            static_cast<std::size_t>(std::sqrt(d)));

    if (feature_search_.mode == FeatureMode::Count)
        return feature_search_.count;

    throw std::invalid_argument(
        "Неизвестный режим выбора признаков");
}

std::optional<Split> CartTree::find_node_split(
    const Matrix &X,
    std::span<const double> y,
    const std::vector<std::size_t> &object_ids,
    const std::vector<std::size_t> &feature_ids,
    std::span<const std::size_t> counts)
{
    const std::size_t q =
        resolve_max_features(feature_ids.size());

    if (feature_search_.mode == FeatureMode::All)
    {
        return finder_.find_best(
            X, y, object_ids, feature_ids, criterion_, min_samples_leaf_, counts);
    }

    std::vector<std::size_t> shuffled_features = feature_ids;

    std::shuffle(
        shuffled_features.begin(),
        shuffled_features.end(),
        rng_);

    std::optional<Split> best_split;
    std::size_t viewed = 0;
    bool has_non_constant = false;

    for (const auto s : shuffled_features)
    {
        auto result = finder_.find_for_feature(
            X, y, object_ids, s, criterion_, min_samples_leaf_, counts);

        viewed++;

        if (!result.is_constant)
        {
            has_non_constant = true;
        }

        if (result.best.has_value())
        {
            bool is_better = false;

            if (!best_split)
            {
                is_better = true;
            }
            else if (result.best->score > best_split->score)
            {
                is_better = true;
            }
            else if (result.best->score == best_split->score)
            {
                if (result.best->feature < best_split->feature)
                {
                    is_better = true;
                }
                else if (result.best->feature == best_split->feature)
                {
                    if (result.best->threshold < best_split->threshold)
                    {
                        is_better = true;
                    }
                }
            }

            if (is_better)
            {
                best_split = result.best;
            }
        }
        if (viewed >= q && has_non_constant)
            break;
    }
    return best_split;
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
    const std::vector<std::size_t> &object_ids,
    std::span<const std::size_t> counts)
{
    auto node = std::make_unique<Node>();

    double count = 0.0;

    for (std::size_t i : object_ids)
    {
        count += counts[i];
    }

    if (criterion_ == SplitCriterion::Variance)
    {
        double mean = 0.0;

        for (std::size_t i : object_ids)
        {
            mean += y[i] * (counts[i] / count);
        }

        if (!std::isfinite(mean))
        {
            throw std::overflow_error(
                "Не удалось вычислить конечное среднее откликов");
        }

        node->value = mean;
    }
    else if (criterion_ == SplitCriterion::Gini)
    {
        node->probabilities.assign(classes_.size(), 0.0);

        for (std::size_t i : object_ids)
        {
            auto it = std::lower_bound(
                classes_.begin(), classes_.end(), y[i]);

            std::size_t c = it - classes_.begin();

            node->probabilities[c] += counts[i];
        }

        for (double &probability : node->probabilities)
        {
            probability /= count;
        }
    }

    if (!node->is_valid())
    {
        throw std::logic_error("Некорректное состояние листа");
    }

    return node;
}

std::unique_ptr<Node> CartTree::build_tree(
    const Matrix &X,
    std::span<const double> y,
    const std::vector<std::size_t> &object_ids,
    const std::vector<std::size_t> &feature_ids,
    std::size_t depth,
    std::span<const std::size_t> counts)
{
    if ((max_depth_.has_value() && depth >= *max_depth_) ||
        object_ids.size() < min_samples_split_)
    {
        return make_leaf(y, object_ids, counts);
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
        return make_leaf(y, object_ids, counts);
    }

    auto split = find_node_split(
        X, y, object_ids, feature_ids, counts);

    if (!split.has_value())
    {
        return make_leaf(y, object_ids, counts);
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
        X, y, left_ids, feature_ids, depth + 1, counts);

    node->right = build_tree(
        X, y, right_ids, feature_ids, depth + 1, counts);

    if (!node->is_valid())
    {
        throw std::logic_error("Некорректное состояние узла");
    }

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

    std::vector<std::size_t> counts(X.rows(), 1);

    fit_with_counts(X, y, counts);
}

void CartTree::fit_with_counts(
    const Matrix &X,
    std::span<const double> y,
    std::span<const std::size_t> counts)
{
    root_.reset();
    n_features_ = 0;
    classes_.clear();

    rng_.seed(random_state_);

    if (X.rows() == 0)
        throw std::invalid_argument("Обучающая выборка пуста");

    if (X.cols() == 0)
        throw std::invalid_argument("Матрица не содержит признаков");

    if (feature_search_.mode == FeatureMode::Count &&
        feature_search_.count > X.cols())
        throw std::invalid_argument(
            "Количество выбираемых признаков превышает число столбцов");

    if (y.size() != X.rows())
        throw std::invalid_argument(
            "Количество строк матрицы X не совпадает с размером y");

    if (counts.size() != X.rows())
        throw std::invalid_argument(
            "Количество элементов в counts не совпадает с количеством строк матрицы X");

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

    bool is_positive = false;
    for (const std::size_t val : counts)
    {
        if (val > 0)
        {
            is_positive = true;
            break;
        }
    }

    if (!is_positive)
        throw std::invalid_argument(
            "Counts должен содержать хотя бы одну положительную кратность");

    if (criterion_ == SplitCriterion::Gini)
    {
        classes_.assign(y.begin(), y.end());

        std::sort(classes_.begin(), classes_.end());

        classes_.erase(
            std::unique(classes_.begin(), classes_.end()),
            classes_.end());
    }

    std::vector<std::size_t> object_ids;

    for (std::size_t i = 0; i < counts.size(); ++i)
    {
        if (counts[i] > 0)
            object_ids.push_back(i);
    }

    std::vector<std::size_t> feature_ids(X.cols());
    std::iota(feature_ids.begin(), feature_ids.end(), 0);

    root_ = build_tree(X, y, object_ids, feature_ids, 0, counts);

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

void CartTree::collect_stats(
    const Node *node,
    std::size_t depth,
    TreeStats &result) const
{
    if (node == nullptr)
    {
        return;
    }

    ++result.node_count;

    result.depth = std::max(result.depth, depth);

    if (node->is_leaf)
    {
        ++result.leaf_count;
        return;
    }

    collect_stats(
        node->left.get(),
        depth + 1,
        result);

    collect_stats(
        node->right.get(),
        depth + 1,
        result);
}

TreeStats CartTree::stats() const
{
    if (!is_fitted())
    {
        throw std::logic_error("Дерево ещё не обучено");
    }

    TreeStats result;

    collect_stats(root_.get(), 0, result);

    return result;
}

void CartTree::collect_splits(
    const Node *node,
    std::size_t depth,
    std::size_t limit,
    std::vector<SplitInfo> &result) const
{
    if (node == nullptr ||
        node->is_leaf ||
        result.size() >= limit)
    {
        return;
    }

    result.push_back({depth,
                      node->feature,
                      node->threshold});

    collect_splits(
        node->left.get(),
        depth + 1,
        limit,
        result);

    collect_splits(
        node->right.get(),
        depth + 1,
        limit,
        result);
}

std::vector<SplitInfo> CartTree::first_splits(
    std::size_t limit) const
{
    if (!is_fitted())
    {
        throw std::logic_error("Дерево ещё не обучено");
    }

    std::vector<SplitInfo> result;

    collect_splits(root_.get(), 0, limit, result);

    return result;
}