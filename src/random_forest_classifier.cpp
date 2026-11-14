#include "random_forest_classifier.hpp"

#include <stdexcept>
#include <cmath>
#include <random>
#include <algorithm>
#include <utility>

RandomForestClassifier::RandomForestClassifier()
    : RandomForestClassifier(
          RandomForestParams{
              .max_features = {FeatureMode::Sqrt, 1}}) // назначенная инициализация
{
}

RandomForestClassifier::RandomForestClassifier(
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