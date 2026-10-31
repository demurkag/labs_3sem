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
    std::size_t min_samples_leaf) const
{
    const std::size_t num_samples = object_ids.size();

    if (X.rows() == 0 || X.cols() == 0)
        throw std::invalid_argument("Матрица X не содержит элементов");

    if (y.size() != X.rows())
        throw std::invalid_argument(
            "Количество строк матрицы X не совпадает с размером y");

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

    if (all_equal)
    {
        return std::nullopt;
    }

    double total_sum = 0.0;

    std::vector<double> classes;                      // уникальные значения
    std::vector<std::size_t> class_indices(X.rows()); // объект -> индекс в классе
    std::vector<std::size_t> class_counts;

    if (criterion == SplitCriterion::Variance)
    {
        for (std::size_t i : object_ids)
        {
            total_sum += y[i];
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

        for (std::size_t i : object_ids)
        {
            const auto it = std::lower_bound(
                classes.begin(), classes.end(), y[i]);

            const std::size_t c = it - classes.begin();

            class_indices[i] = c;
            ++class_counts[c];
        }
    }

    std::vector<std::size_t> order(num_samples);

    std::optional<Split> best_split;

    for (std::size_t j : feature_ids)
    {
        std::copy(object_ids.begin(), object_ids.end(), order.begin());

        std::sort(order.begin(), order.end(),
                  [&X, j](std::size_t a, std::size_t b)
                  {
                      return X(a, j) < X(b, j);
                  });

        double left_sum = 0.0;
        std::size_t left_count = 0;

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
            const std::size_t row = order[i];

            ++left_count;

            if (criterion == SplitCriterion::Variance)
            {
                left_sum += y[row];
            }
            else
            {
                const std::size_t c = class_indices[row];

                const std::size_t a = left_classes[c];
                const std::size_t b = right_classes[c];

                left_squared += 2.0 * a + 1.0;
                right_squared += 1.0 - 2.0 * b;

                ++left_classes[c];
                --right_classes[c];
            }

            const std::size_t right_count = num_samples - left_count;

            if (left_count < min_samples_leaf ||
                right_count < min_samples_leaf)
            {
                continue;
            }

            const double current_value = X(order[i], j);
            const double next_value = X(order[i + 1], j);

            if (current_value == next_value)
                continue;

            double score = 0.0;

            if (criterion == SplitCriterion::Variance)
            {
                const double right_sum = total_sum - left_sum;

                score =
                    left_sum * left_sum / left_count +
                    right_sum * right_sum / right_count;
            }
            else
            {
                score =
                    left_squared / left_count +
                    right_squared / right_count;
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
                if (j < best_split->feature)
                {
                    is_better = true;
                }
                else if (j == best_split->feature)
                {
                    if (threshold < best_split->threshold)
                    {
                        is_better = true;
                    }
                }
            }

            if (is_better)
            {
                best_split = Split{j, threshold, score};
            }
        }
    }

    return best_split;
}