#pragma once

#include "cart_tree.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>

struct RandomForestParams
{
    std::size_t n_estimators = 100;
    bool bootstrap = true;

    std::optional<std::size_t> max_samples = std::nullopt;
    FeatureSearchParams max_features = {};

    std::optional<std::size_t> max_depth = std::nullopt;
    std::size_t min_samples_split = 2;
    std::size_t min_samples_leaf = 1;

    std::uint32_t random_state = 0;
    std::size_t n_jobs = 1;
};