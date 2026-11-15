#pragma once

/** @file random_forest_params.hpp
 * @brief Параметры случайного леса регрессии и классификации.
 */

#include "cart_tree.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>

/** @brief Настройки ансамбля; ограничения проверяют конструктор модели и fit().
 * Значения по умолчанию структуры подходят регрессору. Конструктор
 * RandomForestClassifier() без аргументов отдельно выбирает Sqrt;
 * RandomForestClassifier(params) сохраняет переданный max_features, включая All.
 * Неверные параметры вызывают std::invalid_argument. Ограничения, зависящие
 * от числа строк n и столбцов d, проверяются при fit().
 */
struct RandomForestParams
{
    std::size_t n_estimators = 100; ///< Число деревьев, не меньше 1; по умолчанию 100.
    bool bootstrap = true; ///< Извлечение с возвращением; false задаёт единичные counts.

    /** @brief Число извлечений: nullopt означает n; иначе целое от 1 до n.
     * Заданное значение при bootstrap=false запрещено, включая значение n.
     */
    std::optional<std::size_t> max_samples = std::nullopt;
    FeatureSearchParams max_features = {}; ///< По умолчанию All; правила в FeatureSearchParams.

    std::optional<std::size_t> max_depth = std::nullopt; ///< nullopt: без предела; заданный предел >= 1.
    std::size_t min_samples_split = 2; ///< Минимум разных исходных индексов для попытки, >= 2.
    std::size_t min_samples_leaf = 1; ///< Минимум разных исходных индексов в каждом листе, >= 1.

    std::uint32_t random_state = 0; ///< Общий seed mt19937, по умолчанию 0; любой uint32_t.
    std::size_t n_jobs = 1; ///< Лимит рабочих потоков >= 1; 1 означает последовательный путь.
};
