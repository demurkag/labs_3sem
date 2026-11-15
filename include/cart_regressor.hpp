#pragma once

/** @file cart_regressor.hpp
 * @brief Регрессия CART через интерфейс IRegressor.
 */

#include "imodel.hpp"
#include "cart_tree.hpp"

#include <cstddef>
#include <span>
#include <vector>
#include <cstdint>
#include <optional>

/** @brief Регрессор IRegressor на основе CartTree с критерием Variance.
 * Предсказывает средние откликов в листьях. Владение, рекурсия и остановка
 * описаны в CartTree; в отличие от пня №2 постоянный отклик сразу образует лист.
 * Проверяется конечность входных данных и среднего значения в листьях.
 * Если среднее не является конечным числом, обучение завершается исключением.
 */
class CartRegressor : public IRegressor
{
private:
    CartTree tree_;

public:
    /** @brief Создаёт необученный регрессор.
     * @param max_depth Максимальная глубина в рёбрах; nullopt без предела, 0 один лист.
     * @param min_samples_split Минимум объектов для попытки разбиения, не меньше 2.
     * @param min_samples_leaf Минимум объектов с каждой стороны порога, не меньше 1.
     * @param feature_search Политика признаков в каждом узле, по умолчанию All.
     * @param random_state Seed RNG признаков, по умолчанию 0; сбрасывается при обучении.
     * @throws std::invalid_argument Неверные минимумы, режим признаков или Count=0.
     */
    CartRegressor(
        std::optional<std::size_t> max_depth,
        std::size_t min_samples_split,
        std::size_t min_samples_leaf,
        FeatureSearchParams feature_search = {},
        std::uint32_t random_state = 0);

    /** @brief Обучает дерево, сбрасывая прежнее состояние.
     * @param X Непустая матрица конечных признаков с хотя бы одним столбцом.
     * @param y Конечные отклики, по одному на строку X.
     * @throws std::invalid_argument Пустые размеры, неверная длина y или неконечный вход.
     * @throws std::runtime_error Неконечная оценка разбиения.
     * @throws std::logic_error Нарушена локальная форма построенного узла.
     * @note Успех полностью заменяет дерево; при исключении модель необучена.
     */
    void fit(
        const Matrix &X,
        std::span<const double> y) override;

    /** @brief Обучает CART с bootstrap-кратностями; обычный fit задаёт единицы.
     * @param X Непустая конечная матрица n x d, d >= 1.
     * @param y Ровно n конечных откликов.
     * @param counts Ровно n целых кратностей исходных строк; хотя бы одна >0.
     * Нули исключают индексы. Массив не меняется и не сохраняется по ссылке.
     * @throws std::invalid_argument Неверные размеры, все counts нулевые,
     * неконечный вход или Count>d.
     * @throws std::runtime_error Неконечная оценка разбиения.
     * @throws std::overflow_error Неконечное среднее листа.
     * @throws std::logic_error Нарушена форма узла.
     * @note Критерии и средние взвешены, минимумы размеров считаются по разным
     * исходным индексам. Неудача сбрасывает обучение. См. CartTree::fit_with_counts.
     */
    void fit_with_counts(
        const Matrix &X,
        std::span<const double> y,
        std::span<const std::size_t> counts);

    /** @brief Предсказывает среднее обучающих откликов достигнутого листа.
     * @param X Конечные признаки с обучающим числом и порядком столбцов.
     * @return Один ответ на строку X, пустой вектор для нуля строк.
     * @throws std::logic_error Модель не обучена.
     * @throws std::invalid_argument Неверное число столбцов или неконечный признак.
     */
    std::vector<double> predict(
        const Matrix &X) const override;

    /** @brief Подсчитывает узлы, листья и фактическую глубину.
     * @return Статистика CartTree::stats().
     * @throws std::logic_error Модель не обучена.
     */
    TreeStats stats() const;

    /** @brief Возвращает первые внутренние узлы в прямом обходе дерева.
     * @param limit Максимум записей; 0 допустим.
     * @return Разбиения в порядке узел–левое–правое, без листьев.
     * @throws std::logic_error Модель не обучена.
     */
    std::vector<SplitInfo> first_splits(
        std::size_t limit) const;
};
