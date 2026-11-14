#include "split_finder.hpp"

#include <vector>
#include <optional>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <stdexcept>

std::optional<Split> SplitFinder::find_best(
    const Matrix &X,
    std::span<const double> y,
    std::span<const std::size_t> object_ids,
    std::span<const std::size_t> feature_ids,
    SplitCriterion criterion,
    std::size_t min_samples_leaf,
    std::span<const std::size_t> counts) const
{
    const std::size_t num_samples = object_ids.size();

    if (X.rows() == 0 || X.cols() == 0)
        throw std::invalid_argument("Матрица X не содержит элементов");

    if (y.size() != X.rows())
        throw std::invalid_argument(
            "Количество строк матрицы X не совпадает с размером y");

    if (!counts.empty() && counts.size() != X.rows())
        throw std::invalid_argument(
            "Вектор кратностей должен быть пустым или количество элементов должно совпадать с количеством строк матрицы X");

    if (num_samples == 0)
        throw std::invalid_argument("Список объектов пуст");

    if (min_samples_leaf == 0)
        throw std::invalid_argument(
            "min_samples_leaf должен быть больше нуля");

    if (criterion != SplitCriterion::Variance &&
        criterion != SplitCriterion::Gini)
        throw std::invalid_argument("Неизвестный критерий");

    for (std::size_t i : object_ids)
    {
        if (i >= X.rows())
            throw std::out_of_range("Некорректный индекс объекта");

        if (!counts.empty() && counts[i] == 0)
            throw std::invalid_argument(
                "Кратность объекта должна быть положительной");
    }

    for (std::size_t j : feature_ids)
    {
        if (j >= X.cols())
            throw std::out_of_range("Некорректный индекс признака");
    }

    for (std::size_t i : object_ids)
    {
        if (!std::isfinite(y[i]))
            throw std::invalid_argument("Неконечный отклик");

        for (std::size_t j : feature_ids)
        {
            if (!std::isfinite(X(i, j)))
                throw std::invalid_argument("Неконечное значение признака");
        }
    }

    if (feature_ids.empty() || num_samples / 2 < min_samples_leaf)
    {
        return std::nullopt;
    }

    const double first_value = y[object_ids[0]];

    bool all_equal = true;

    for (std::size_t i : object_ids)
    {
        if (y[i] != first_value)
        {
            all_equal = false;
            break;
        }
    }

    if (criterion == SplitCriterion::Gini && all_equal)
    {
        return std::nullopt;
    }

    std::optional<Split> best_split;

    for (std::size_t j : feature_ids)
    {
        auto result = find_for_feature(
            X,
            y,
            object_ids,
            j,
            criterion,
            min_samples_leaf,
            counts);

        if (!result.best.has_value())
            continue;

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

    return best_split;
}

FeatureSplitResult SplitFinder::find_for_feature(
    const Matrix &X,
    std::span<const double> y,
    std::span<const std::size_t> object_ids,
    std::size_t feature,
    SplitCriterion criterion,
    std::size_t min_samples_leaf,
    std::span<const std::size_t> counts) const
{
    const std::size_t num_samples = object_ids.size();

    if (X.rows() == 0 || X.cols() == 0)
        throw std::invalid_argument("Матрица X не содержит элементов");

    if (y.size() != X.rows())
        throw std::invalid_argument(
            "Количество строк матрицы X не совпадает с размером y");

    if (!counts.empty() && counts.size() != X.rows())
        throw std::invalid_argument(
            "Counts должен быть пустым или количество элементов должно совпадать с количеством строк матрицы X");

    if (num_samples == 0)
        throw std::invalid_argument("Список объектов пуст");

    if (min_samples_leaf == 0)
        throw std::invalid_argument(
            "min_samples_leaf должен быть больше нуля");

    if (criterion != SplitCriterion::Variance &&
        criterion != SplitCriterion::Gini)
        throw std::invalid_argument("Неизвестный критерий");

    if (feature >= X.cols())
        throw std::out_of_range("Некорректный индекс признака");

    for (std::size_t i : object_ids)
    {
        if (i >= X.rows())
            throw std::out_of_range("Некорректный индекс объекта");

        if (!counts.empty() && counts[i] == 0)
            throw std::invalid_argument(
                "Кратность объекта должна быть положительной");
    }

    for (std::size_t i : object_ids)
    {
        if (!std::isfinite(y[i]))
            throw std::invalid_argument("Неконечный отклик");

        if (!std::isfinite(X(i, feature)))
            throw std::invalid_argument("Неконечное значение признака");
    }

    std::vector<std::size_t> order(num_samples);

    std::optional<Split> best_split;

    std::iota(order.begin(), order.end(), 0);

    std::sort(order.begin(), order.end(),
              [&X, &object_ids, feature](std::size_t a, std::size_t b)
              {
                  return X(object_ids[a], feature) <
                         X(object_ids[b], feature);
              });

    const bool is_constant =
        X(object_ids[order.front()], feature) ==
        X(object_ids[order.back()], feature);

    if (is_constant)
    {
        return {true, std::nullopt};
    }

    if (num_samples / 2 < min_samples_leaf)
    {
        return {false, std::nullopt};
    }

    const double first_value = y[object_ids[0]];

    bool all_equal = true;

    for (std::size_t i : object_ids)
    {
        if (y[i] != first_value)
        {
            all_equal = false;
            break;
        }
    }

    if (criterion == SplitCriterion::Gini && all_equal)
    {
        return {false, std::nullopt};
    }

    double total_sum = 0.0;
    double total_weight = 0.0;

    std::vector<double> classes;
    std::vector<std::size_t> class_indices;
    std::vector<std::size_t> class_counts;

    if (criterion == SplitCriterion::Variance)
    {
        for (std::size_t i : object_ids)
        {
            const std::size_t w = counts.empty() ? 1 : counts[i];

            total_weight += w;
            total_sum += y[i] * w;
        }
    }
    else
    {
        for (std::size_t i : object_ids)
        {
            classes.push_back(y[i]);
        }

        std::sort(classes.begin(), classes.end());

        classes.erase(
            std::unique(classes.begin(), classes.end()),
            classes.end());

        class_counts.assign(classes.size(), 0);

        class_indices.reserve(num_samples);

        for (std::size_t i : object_ids)
        {
            const auto it = std::lower_bound(
                classes.begin(), classes.end(), y[i]);

            const std::size_t c = it - classes.begin();
            const std::size_t w = counts.empty() ? 1 : counts[i];

            class_indices.push_back(c);
            class_counts[c] += w;
            total_weight += w;
        }
    }

    double left_sum = 0.0;
    std::size_t left_count = 0;
    double left_weight = 0.0;

    std::vector<std::size_t> left_classes(classes.size(), 0);
    std::vector<std::size_t> right_classes = class_counts;

    double left_squared = 0.0;
    double right_squared = 0.0;

    if (criterion == SplitCriterion::Gini)
    {
        for (std::size_t count : class_counts)
        {
            right_squared += static_cast<double>(count) * count;
        }
    }

    for (std::size_t i = 0; i + 1 < num_samples; ++i)
    {
        const std::size_t local_index = order[i];

        const std::size_t row = object_ids[local_index];

        const std::size_t w = counts.empty() ? 1 : counts[row];

        ++left_count;
        left_weight += w;

        if (criterion == SplitCriterion::Variance)
        {
            left_sum += y[row] * w;
        }
        else
        {
            const std::size_t c = class_indices[local_index];

            const std::size_t a = left_classes[c];
            const std::size_t b = right_classes[c];

            left_squared += 2.0 * a * w + static_cast<double>(w) * w;
            right_squared += -2.0 * b * w + static_cast<double>(w) * w;

            left_classes[c] += w;
            right_classes[c] -= w;
        }

        const std::size_t right_count = num_samples - left_count;
        const double right_weight = total_weight - left_weight;

        if (left_count < min_samples_leaf ||
            right_count < min_samples_leaf)
        {
            continue;
        }

        const double current_value = X(object_ids[order[i]], feature);
        const double next_value = X(object_ids[order[i + 1]], feature);

        if (current_value == next_value)
            continue;

        double score = 0.0;

        if (criterion == SplitCriterion::Variance)
        {
            const double right_sum = total_sum - left_sum;

            score =
                left_sum * left_sum / left_weight +
                right_sum * right_sum / right_weight;
        }
        else
        {
            score =
                left_squared / left_weight +
                right_squared / right_weight;
        }

        if (!std::isfinite(score))
            throw std::runtime_error("Неконечная оценка разбиения");

        double threshold = std::midpoint(current_value, next_value);

        if (threshold == next_value)
            threshold = current_value;

        bool is_better = false;

        if (!best_split)
        {
            is_better = true;
        }
        else if (score > best_split->score)
        {
            is_better = true;
        }
        else if (score == best_split->score)
        {
            if (threshold < best_split->threshold)
                is_better = true;
        }

        if (is_better)
        {
            best_split = Split{feature, threshold, score};
        }
    }

    return {false, best_split};
}
