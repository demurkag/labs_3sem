#pragma once

/** @file cart_classifier.hpp
 * @brief Классификация CART через интерфейс IClassifier.
 */

#include "imodel.hpp"
#include "cart_tree.hpp"

#include <cstddef>
#include <span>
#include <vector>
#include <cstdint>
#include <optional>

/** @brief Классификатор IClassifier на основе CartTree с критерием Gini.
 * Конструктор не обучает модель. Остановка, владение узлами и общий поиск
 * разбиений описаны в CartTree; печать результатов выполняет приложение.
 */
class CartClassifier : public IClassifier
{
private:
    CartTree tree_;

public:
    /** @brief Создаёт необученный классификатор.
     * @param max_depth Максимальная глубина в рёбрах; nullopt без предела, 0 один лист.
     * @param min_samples_split Минимум объектов для попытки разбиения, не меньше 2.
     * @param min_samples_leaf Минимум объектов с каждой стороны порога, не меньше 1.
     * @param feature_search Политика признаков в каждом узле, по умолчанию All.
     * @param random_state Seed RNG признаков, по умолчанию 0; сбрасывается при обучении.
     * @throws std::invalid_argument Неверные минимумы, режим признаков или Count=0.
     */
    CartClassifier(
        std::optional<std::size_t> max_depth,
        std::size_t min_samples_split,
        std::size_t min_samples_leaf,
        FeatureSearchParams feature_search = {},
        std::uint32_t random_state = 0);

    /** @brief Обучает модель и заменяет прежнее состояние через CartTree::fit().
     * @param X Непустая матрица конечных признаков с хотя бы одним столбцом.
     * @param y Конечная исходная метка класса для каждой строки X.
     * @throws std::invalid_argument Пустые размеры, неверная длина y или неконечный вход.
     * @throws std::runtime_error Неконечная оценка разбиения.
     * @throws std::logic_error Нарушена локальная форма построенного узла.
     * @note Неуспешный fit оставляет модель необученной. Порядок классов — возрастающий.
     */
    void fit(
        const Matrix &X,
        std::span<const double> y) override;

    /** @brief Обучает CART с целочисленными bootstrap-кратностями.
     * @param X Непустая конечная матрица n x d, d >= 1.
     * @param y Ровно n конечных исходных меток; classes включает всё y.
     * @param counts Ровно n кратностей исходных строк, хотя бы одна >0;
     * нули исключают индексы, отсутствующему классу соответствует вероятность 0.
     * Массив не изменяется и не сохраняется по ссылке.
     * @throws std::invalid_argument Неверные размеры, все counts нулевые,
     * неконечный вход или Count>d.
     * @throws std::runtime_error Неконечная оценка разбиения.
     * @throws std::logic_error Нарушена форма узла.
     * @note Gini и частоты взвешены, минимумы размеров считаются по разным
     * исходным индексам. Неудача сбрасывает обучение. См. CartTree::fit_with_counts.
     */
    void fit_with_counts(
        const Matrix &X,
        std::span<const double> y,
        std::span<const std::size_t> counts);

    /** @brief Возвращает исходные метки наиболее вероятных классов.
     * @param X Конечные признаки с тем же числом и порядком столбцов, что при fit().
     * @return Метка для каждой строки; для нуля строк пустой вектор.
     * При равенстве максимальных вероятностей выбирается меньшая метка.
     * @throws std::logic_error Модель не обучена.
     * @throws std::invalid_argument Неверное число столбцов или неконечный признак.
     */
    std::vector<double> predict(
        const Matrix &X) const override;

    /** @brief Возвращает взвешенные частоты классов в достигнутых листьях.
     * @param X Конечные признаки с обучающим числом и порядком столбцов.
     * @return Матрица RowMajor X.rows() x classes().size(); столбец c соответствует
     * classes()[c]. Отсутствующий в листе класс имеет вероятность 0.
     * Ноль строк допустим: результат 0 x число классов.
     * @throws std::logic_error Модель не обучена.
     * @throws std::invalid_argument Неверное число столбцов или неконечный признак.
     */
    Matrix predict_proba(
        const Matrix &X) const override;

    /** @brief Возвращает исходные метки классов по возрастанию.
     * @return Невладеющий список, действующий до следующего fit() или уничтожения модели.
     * @throws std::logic_error Модель не обучена.
     */
    std::span<const double> classes() const override;

    /** @brief Подсчитывает узлы, листья и фактическую глубину.
     * @return Статистика CartTree::stats().
     * @throws std::logic_error Модель не обучена.
     */
    TreeStats stats() const;

    /** @brief Возвращает первые внутренние узлы в прямом обходе дерева.
     * @param limit Максимальное число записей; 0 допустим.
     * @return Разбиения в порядке узел–левое–правое, без листьев.
     * @throws std::logic_error Модель не обучена.
     */
    std::vector<SplitInfo> first_splits(
        std::size_t limit) const;
};
