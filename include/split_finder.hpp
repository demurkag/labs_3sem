#pragma once

/** @file split_finder.hpp
 * @brief Общий поиск разбиения для решающего пня и CART.
 */

#include "enums.hpp"
#include "matrix.hpp"

#include <cstddef>
#include <optional>
#include <span>

/** @brief Выбранное разбиение и максимизируемая оценка. */
struct Split // оптимальное разбиение
{
    std::size_t feature; ///< Индекс столбца исходной X.
    double threshold;    ///< Значение для маршрутизации X(i, feature) <= threshold влево.
    double score;        ///< Сумма по сторонам: Variance S^2/W, Gini sum(n_c^2)/W;
                         ///< S и n_c взвешены counts, W — сумма кратностей.
};

/** @brief Результат поиска на одном признаке, включая его константность. */
struct FeatureSplitResult {
    bool is_constant; ///< Все выбранные объекты имеют точно равные значения признака.
    std::optional<Split> best; ///< Лучший допустимый порог; nullopt не обязательно означает константность.
};

/** @brief Ищет лучший порог без копирования подматрицы.
 * Сортирует индексы объектов для каждого разрешённого признака и обновляет
 * агрегаты при проходе по границам. Пень и дерево используют одну реализацию.
 * Для s выбранных объектов и d признаков поиск занимает
 * O(d * s * log(s)) при s >= 2.
 * В режиме Gini индексы классов хранятся только для объектов
 * текущего узла; дополнительная память имеет порядок O(s).
 */
class SplitFinder
{
public:
    /** @brief Выбирает допустимую границу между разными соседними значениями.
     * @param X Исходная непустая матрица с обоими поддерживаемыми layout.
     * @param y Отклики всех строк X; для Gini — числовые метки классов.
     * @param object_ids Непустой набор разных исходных индексов с положительными counts.
     * @param feature_ids Индексы разрешённых столбцов; остальные не рассматриваются.
     * @param criterion Variance для регрессии, Gini для классификации.
     * @param min_samples_leaf Положительный минимум объектов с каждой стороны.
     * @param counts Пустой span означает единицы; иначе ровно X.rows() кратностей,
     * индексируемых исходной строкой. Вне object_ids нули допустимы.
     * @return Разбиение с наибольшим score либо nullopt, если нет разрешённых
     * признаков, допустимых размеров/границ или при одном классе Gini.
     * Постоянный отклик Variance сам по себе не запрещает разбиение.
     * @throws std::invalid_argument Пустая X/набор объектов, неверный размер y,
     * неизвестный критерий, нулевой минимум листа, неверный размер counts,
     * нулевой вес выбранного объекта или неконечные выбранные X/y.
     * @throws std::out_of_range Индекс строки или столбца вне X.
     * @throws std::runtime_error Вычисленная оценка не конечна.
     * @note Порог — std::midpoint соседей; если округлён до верхнего соседа,
     * заменяется нижним для согласования с <=. При строго равном double-score
     * выбирается меньший индекс признака, затем меньший порог. Сравнения точные.
     * Проверки улучшения относительно родительского узла нет.
     * Конечность входа не предотвращает переполнение промежуточных сумм/квадратов.
     */
    std::optional<Split> find_best(
        const Matrix &X,
        std::span<const double> y,
        std::span<const std::size_t> object_ids,
        std::span<const std::size_t> feature_ids,
        SplitCriterion criterion,
        std::size_t min_samples_leaf,
        std::span<const std::size_t> counts = {}) const;

    /** @brief Ищет лучший порог одного столбца и отдельно сообщает константность.
     * @param X Непустая исходная матрица с хотя бы одним столбцом; оба layout.
     * @param y Конечные отклики выбранных строк, размер X.rows().
     * @param object_ids Непустой набор разных исходных индексов с положительными counts.
     * @param feature Индекс единственного рассматриваемого столбца.
     * @param criterion Variance либо Gini.
     * @param min_samples_leaf Минимум разных исходных индексов с каждой стороны, >=1.
     * @param counts Пусто — единицы; иначе X.rows() кратностей по исходной строке.
     * @return is_constant определяется до проверки размеров частей и числа
     * классов. best отсутствует при константном признаке, недопустимых размерах
     * либо одном классе Gini; иначе содержит максимум score.
     * @throws std::invalid_argument Неверные размеры/критерий/минимум,
     * нулевой вес выбранного объекта или неконечные выбранные X/y.
     * @throws std::out_of_range Индекс строки или признака вне X.
     * @throws std::runtime_error Неконечный score.
     * @note Веса влияют на score, но не на подсчёт min_samples_leaf.
     * Правила порога и точных ничьих те же, что у find_best.
     */
    FeatureSplitResult find_for_feature(
        const Matrix &X,
        std::span<const double> y,
        std::span<const std::size_t> object_ids,
        std::size_t feature,
        SplitCriterion criterion,
        std::size_t min_samples_leaf,
        std::span<const std::size_t> counts = {}) const;
};
