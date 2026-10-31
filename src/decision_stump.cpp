#include "decision_stump.hpp"
#include "split_finder.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

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

void validate_finite_features(const Matrix &features)
{
    for (std::size_t i = 0; i < features.rows(); ++i)
    {
        for (std::size_t j = 0; j < features.cols(); ++j)
        {
            if (!std::isfinite(features(i, j)))
                throw std::invalid_argument("Матрица содержит NaN или бесконечность");
        }
    }
}

void DecisionStump::fit(const Matrix &features,
                        std::span<const double> target)
{
    is_fitted_ = false;
    has_split_ = false;

    const std::size_t num_samples = features.rows();
    const std::size_t num_features = features.cols();

    if (num_samples == 0)
        throw std::invalid_argument("Обучающая выборка пуста");

    if (num_features == 0)
        throw std::invalid_argument("Матрица не содержит признаков");

    if (target.size() != num_samples)
        throw std::invalid_argument(
            "Количество строк features не совпадает с размером target");

    for (double value : target)
    {
        if (!std::isfinite(value))
            throw std::invalid_argument("Отклик содержит NaN или бесконечность");
    }
    validate_finite_features(features);

    std::vector<std::size_t> object_ids(num_samples);
    std::vector<std::size_t> feature_ids(num_features);

    std::iota(object_ids.begin(), object_ids.end(), 0);
    std::iota(feature_ids.begin(), feature_ids.end(), 0);

    SplitFinder finder;
    auto best_split = finder.find_best(features, target, object_ids, feature_ids,
                                       criterion_, min_samples_leaf_);

    has_split_ = best_split.has_value();

    if (has_split_)
    {
        feature_index_ = best_split->feature;
        threshold_ = best_split->threshold;
    }

    double total_sum = 0.0;

    std::vector<double> classes;
    std::vector<std::size_t> class_indices(num_samples);
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

    validate_finite_features(features);

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
