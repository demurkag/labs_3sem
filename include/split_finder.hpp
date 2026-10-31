#pragma once

#include "enums.hpp"
#include "matrix.hpp"

#include <cstddef>
#include <optional>
#include <span>

struct Split // оптимальное разбиение
{
    std::size_t feature;
    double threshold;
    double score;
};

class SplitFinder
{
public:
    std::optional<Split> find_best(
        const Matrix &X,
        std::span<const double> y,
        std::span<const std::size_t> object_ids,
        std::span<const std::size_t> feature_ids,
        SplitCriterion criterion,
        std::size_t min_samples_leaf) const;
};