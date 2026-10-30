#include "decision_stump.hpp"

#include <stdexcept>
#include <vector>
#include <numeric>
#include <algorithm>
#include <limits>

double majority_class(
    const std::vector<std::size_t> &counts,
    const std::vector<double> &classes)
{
    std::size_t best = 0;

    for (std::size_t c = 1; c < counts.size(); ++c)
    {
        if (counts[c] > counts[best])
        {
            best = c;
        }
    }

    return classes[best];
}

void DecisionStump::fit(const Matrix &features,
                        std::span<const double> target)
{
    const std::size_t num_samples = features.rows();
    const std::size_t num_features = features.cols();

    if (num_samples == 0)
        throw std::invalid_argument("Обучающая выборка пуста");

    if (num_features == 0)
        throw std::invalid_argument("Матрица не содержит признаков");

    if (target.size() != num_samples)
        throw std::invalid_argument(
            "Количество строк features не совпадает с размером target");

    double total_sum = 0.0;

    std::vector<double> classes;                         // уникальные значения
    std::vector<std::size_t> class_indices(num_samples); // объект -> индекс в классе
    std::vector<std::size_t> class_counts;

    if (criterion_ == SplitCriterion::Variance)
    {
        for (double value : target)
        {
            total_sum += value;
        }
    }
    else if (criterion_ == SplitCriterion::Gini)
    {
        classes.assign(target.begin(), target.end());

        std::sort(classes.begin(), classes.end());
        classes.erase(
            std::unique(classes.begin(), classes.end()),
            classes.end());

        class_counts.assign(classes.size(), 0);

        for (std::size_t i = 0; i < num_samples; ++i)
        {
            const auto it = std::lower_bound(
                classes.begin(), classes.end(), target[i]);

            const std::size_t c = it - classes.begin();

            class_indices[i] = c;
            ++class_counts[c];
        }
    }

    if (criterion_ == SplitCriterion::Gini && classes.size() == 1)
    {
        has_split_ = false;
        left_value_ = classes[0];
        right_value_ = classes[0];

        n_features_ = num_features;
        is_fitted_ = true;
        return;
    }

    std::vector<std::size_t> order(num_samples);

    double best_score = -std::numeric_limits<double>::infinity();

    has_split_ = false;

    for (std::size_t j = 0; j < num_features; ++j)
    {
        std::iota(order.begin(), order.end(), 0);

        std::sort(order.begin(), order.end(),
                  [&features, j](std::size_t a, std::size_t b)
                  {
                      return features(a, j) < features(b, j);
                  });

        double left_sum = 0.0;
        std::size_t left_count = 0;

        std::vector<std::size_t> left_classes(classes.size(), 0);
        std::vector<std::size_t> right_classes = class_counts;

        double left_squared = 0.0;
        double right_squared = 0.0;

        if (criterion_ == SplitCriterion::Gini)
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

            if (criterion_ == SplitCriterion::Variance)
            {
                left_sum += target[row];
            }
            else if (criterion_ == SplitCriterion::Gini)
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

            if (left_count < min_samples_leaf_ || right_count < min_samples_leaf_)
            {
                continue;
            }

            const double current_value = features(order[i], j);
            const double next_value = features(order[i + 1], j);

            if (current_value == next_value)
            {
                continue;
            }

            double score = 0.0;

            if (criterion_ == SplitCriterion::Variance)
            {
                const double right_sum = total_sum - left_sum;

                score =
                    left_sum * left_sum / left_count +
                    right_sum * right_sum / right_count;
            }
            else if (criterion_ == SplitCriterion::Gini)
            {
                score =
                    left_squared / left_count +
                    right_squared / right_count;
            }

            if (!has_split_ || score > best_score)
            {
                best_score = score;
                feature_index_ = j;
                threshold_ = std::midpoint(current_value, next_value);
                has_split_ = true;
            }
        }
    }

    if (criterion_ == SplitCriterion::Variance)
    {
        if (!has_split_)
        {
            left_value_ = total_sum / num_samples;
            right_value_ = left_value_;
        }
        else
        {
            double left_sum = 0.0;
            double right_sum = 0.0;

            std::size_t left_count = 0;
            std::size_t right_count = 0;

            for (std::size_t i = 0; i < num_samples; ++i)
            {
                if (features(i, feature_index_) <= threshold_)
                {
                    left_sum += target[i];
                    ++left_count;
                }
                else
                {
                    right_sum += target[i];
                    ++right_count;
                }
            }

            left_value_ = left_sum / left_count;
            right_value_ = right_sum / right_count;
        }
    }
    else if (criterion_ == SplitCriterion::Gini)
    {
        if (!has_split_)
        {
            left_value_ = majority_class(class_counts, classes);
            right_value_ = left_value_;
        }
        else
        {
            std::vector<std::size_t> left_classes(classes.size(), 0);
            std::vector<std::size_t> right_classes(classes.size(), 0);

            for (std::size_t i = 0; i < num_samples; ++i)
            {
                const std::size_t c = class_indices[i];

                if (features(i, feature_index_) <= threshold_)
                {
                    ++left_classes[c];
                }
                else
                {
                    ++right_classes[c];
                }
            }

            left_value_ = majority_class(left_classes, classes);
            right_value_ = majority_class(right_classes, classes);
        }
    }

    n_features_ = num_features;
    is_fitted_ = true;
}

std::vector<double> DecisionStump::predict(const Matrix &features) const
{
    if (!is_fitted_)
    {
        throw std::logic_error("Модель ещё не обучена");
    }

    if (features.cols() != n_features_)
    {
        throw std::invalid_argument(
            "Количество признаков не совпадает с обучающей выборкой");
    }

    std::vector<double> predictions(features.rows());

    for (std::size_t i = 0; i < features.rows(); ++i)
    {
        if (!has_split_ || features(i, feature_index_) <= threshold_)
        {
            predictions[i] = left_value_;
        }
        else
        {
            predictions[i] = right_value_;
        }
    }

    return predictions;
}
