#pragma once

/** @file cart_regressor.hpp
 * @brief Регрессия CART через интерфейс IRegressor.
 */

#include "imodel.hpp"
#include "cart_tree.hpp"

#include <cstddef>
#include <span>
#include <vector>

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
     * @param max_depth Максимальная глубина в рёбрах, 0 означает один лист.
     * @param min_samples_split Минимум объектов для попытки разбиения, не меньше 2.
     * @param min_samples_leaf Минимум объектов с каждой стороны порога, не меньше 1.
     * @throws std::invalid_argument Недопустимые минимумы объектов.
     */
    CartRegressor(
        std::size_t max_depth,
        std::size_t min_samples_split,
        std::size_t min_samples_leaf);

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
